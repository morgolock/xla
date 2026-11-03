/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/pjrt/infer_dispatch_info.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/pjrt/dynamic_shapes.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/utils.h"
#include "xla/service/computation_layout.h"
#include "xla/service/device_assignment.h"
#include "xla/service/hlo_value.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

std::vector<Shape> GetParameterShapes(const ComputationLayout& layout) {
  // For now, XLA programs compiled with multiple arguments for PJRT cannot use
  // tuples for any of their arguments, so we can assume that a tuple can only
  // arise when there is a single argument.
  std::vector<Shape> shapes;
  if (layout.parameter_count() == 1 && layout.parameter_shape(0).IsTuple()) {
    shapes.reserve(layout.parameter_shape(0).tuple_shapes().size());
    absl::c_copy(layout.parameter_shape(0).tuple_shapes(),
                 std::back_inserter(shapes));
  } else {
    shapes.reserve(layout.parameter_count());
    for (const ShapeLayout& sl : layout.parameter_layouts()) {
      shapes.push_back(sl.shape());
    }
  }
  return shapes;
}

namespace {

bool IsPadRealToStaticOp(mlir::Operation* op) {
  auto attr = op->getAttrOfType<mlir::StringAttr>("call_target_name");
  return attr != nullptr && attr.getValue() == "PadRealToStatic";
}

struct OperandSlot {
  const HloInstruction* instruction;
  int64_t operand_number;
};

bool ShapeIndexOverlaps(const ShapeIndex& a, const ShapeIndex& b) {
  const size_t min_size = std::min(a.size(), b.size());
  for (size_t i = 0; i < min_size; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

bool HasOutputToOperandAlias(const HloInstruction* instruction,
                             int64_t operand_number,
                             const ShapeIndex& operand_index) {
  return absl::c_any_of(
      instruction->output_operand_aliasing(), [&](const auto& alias) {
        return alias.second.first == operand_number &&
               ShapeIndexOverlaps(alias.second.second, operand_index);
      });
}

bool HasMatchingCallSignature(const HloInstruction* instruction,
                              const HloComputation* callee) {
  if (callee == nullptr || callee->root_instruction() == nullptr ||
      instruction->operand_count() != callee->num_parameters() ||
      !ShapeUtil::Compatible(instruction->shape(),
                             callee->root_instruction()->shape())) {
    return false;
  }
  for (int64_t i = 0; i < instruction->operand_count(); ++i) {
    if (!ShapeUtil::Compatible(instruction->operand(i)->shape(),
                               callee->parameter_instruction(i)->shape())) {
      return false;
    }
  }
  return true;
}

// Matches root embedded `kCustomCall` dispatchers of offloaded async wrappers
// that forward operands positionally (where `HloDataflowAnalysis` stops).
bool IsAsyncWrappedSubcomputationCall(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kCustomCall ||
      instruction->IsCustomCall("PadRealToStatic") ||
      instruction->called_computations().size() != 1) {
    return false;
  }
  const HloComputation* wrapper = instruction->parent();
  if (wrapper == nullptr || wrapper->root_instruction() != instruction ||
      wrapper->execution_thread() == HloInstruction::kMainExecutionThread ||
      !HasMatchingCallSignature(instruction, instruction->to_apply())) {
    return false;
  }
  const absl::InlinedVector<HloInstruction*, 1> all_callers =
      wrapper->caller_instructions();
  const absl::InlinedVector<HloInstruction*, 1> start_callers =
      wrapper->caller_instructions(HloOpcode::kAsyncStart);
  return !start_callers.empty() &&
         absl::c_all_of(all_callers, [&](const HloInstruction* caller) {
           return caller->IsAsynchronous() &&
                  caller->async_wrapped_instruction() == instruction;
         });
}

// Maps each parameter of `async_inst`'s wrapped computation to its bound slot.
std::vector<OperandSlot> GetAsyncBoundOperandSlots(
    const HloInstruction* async_inst) {
  const HloInstruction* start_inst = async_inst->async_chain_start();
  const HloAsyncStartInstruction* start =
      start_inst != nullptr ? DynCast<HloAsyncStartInstruction>(start_inst)
                            : nullptr;
  const HloInstruction* done = async_inst->async_chain_done();
  const HloComputation* callee = async_inst->async_wrapped_computation();
  if (start == nullptr || done == nullptr || callee == nullptr) {
    return {};
  }
  std::vector<OperandSlot> slots;
  slots.reserve(callee->num_parameters());
  bool reached_done = false;
  for (const HloAsyncInstruction* inst : start->GetAsyncChain()) {
    const int64_t first_operand =
        inst->opcode() == HloOpcode::kAsyncStart ? 0 : 1;
    for (int64_t i = first_operand; i < inst->operand_count(); ++i) {
      const HloInstruction* operand = inst->operand(i);
      if (operand == nullptr ||
          slots.size() >= static_cast<size_t>(callee->num_parameters()) ||
          !ShapeUtil::Compatible(
              operand->shape(),
              callee->parameter_instruction(slots.size())->shape())) {
        return {};
      }
      slots.push_back({inst, i});
    }
    if (inst == done) {
      reached_done = true;
      break;
    }
  }
  if (!reached_done ||
      slots.size() != static_cast<size_t>(callee->num_parameters())) {
    return {};
  }
  return slots;
}

const HloInstruction* GetSubcomputationCallerOperand(
    const HloInstruction* caller, const HloComputation* callee,
    int64_t parameter_number) {
  if (parameter_number < 0 || parameter_number >= callee->num_parameters()) {
    return nullptr;
  }
  if (caller->opcode() == HloOpcode::kCall ||
      IsAsyncWrappedSubcomputationCall(caller)) {
    if (caller->to_apply() != callee ||
        !HasMatchingCallSignature(caller, callee)) {
      return nullptr;
    }
    return caller->operand(parameter_number);
  }
  if (caller->opcode() == HloOpcode::kAsyncStart &&
      caller->async_wrapped_computation() == callee) {
    const std::vector<OperandSlot> slots = GetAsyncBoundOperandSlots(caller);
    if (parameter_number < static_cast<int64_t>(slots.size())) {
      const OperandSlot& slot = slots[parameter_number];
      return slot.instruction->operand(slot.operand_number);
    }
  }
  return nullptr;
}

const HloInstruction* GetUnaliasedForwardedParameter(const HloUse& use) {
  const HloInstruction* user = use.instruction;
  if (user->opcode() == HloOpcode::kCall ||
      IsAsyncWrappedSubcomputationCall(user)) {
    const HloComputation* callee = user->to_apply();
    if (!HasMatchingCallSignature(user, callee) || use.operand_number < 0 ||
        use.operand_number >= callee->num_parameters() ||
        HasOutputToOperandAlias(user, use.operand_number, use.operand_index)) {
      return nullptr;
    }
    return callee->parameter_instruction(use.operand_number);
  }
  if (user->opcode() == HloOpcode::kAsyncStart ||
      (user->opcode() == HloOpcode::kAsyncUpdate && use.operand_number > 0)) {
    const std::vector<OperandSlot> slots = GetAsyncBoundOperandSlots(user);
    for (size_t param_no = 0; param_no < slots.size(); ++param_no) {
      if (slots[param_no].instruction == user &&
          slots[param_no].operand_number == use.operand_number) {
        if (HasOutputToOperandAlias(user, use.operand_number,
                                    use.operand_index) ||
            HasOutputToOperandAlias(user->async_chain_start(),
                                    static_cast<int64_t>(param_no),
                                    use.operand_index)) {
          return nullptr;
        }
        return user->async_wrapped_computation()->parameter_instruction(
            param_no);
      }
    }
  }
  return nullptr;
}

// Returns true if `use` is the pad itself, a bitcast of the buffer, or a
// positional call / async boundary whose callee parameter uses are verified.
bool IsAllowedPadRealToStaticUse(
    const HloDataflowAnalysis& dataflow,
    const absl::flat_hash_set<HloValue::Id>& verified_value_ids,
    const HloValue& value, const HloUse& use) {
  const HloInstruction* user = use.instruction;
  if (user->IsCustomCall("PadRealToStatic")) {
    return use.operand_number == 0 && use.operand_index.empty();
  }
  if (user->opcode() == HloOpcode::kBitcast) {
    return true;
  }
  if ((user->opcode() == HloOpcode::kAsyncUpdate ||
       user->opcode() == HloOpcode::kAsyncDone) &&
      use.operand_number == 0) {
    // Dataflow keeps bound async inputs in context slot {0} across
    // `async-update` until `async-done` consumes the context tuple.
    return user->async_chain_start() != nullptr && !use.operand_index.empty() &&
           use.operand_index.front() == 0 &&
           (user->opcode() != HloOpcode::kAsyncUpdate ||
            !HasOutputToOperandAlias(user, 0, use.operand_index));
  }
  const HloInstruction* param = GetUnaliasedForwardedParameter(use);
  if (param == nullptr ||
      !ShapeUtil::IndexIsValid(param->shape(), use.operand_index)) {
    return false;
  }
  const std::vector<const HloValue*>& param_values =
      dataflow.GetValueSet(param, use.operand_index).values();
  if (absl::c_linear_search(param_values, &value)) {
    return param->user_count() > 0 &&
           absl::c_any_of(value.GetUses(), [&](const HloUse& u) {
             return u.instruction->parent() == param->parent();
           });
  }
  return !param_values.empty() &&
         absl::c_all_of(param_values, [&](const HloValue* v) {
           return verified_value_ids.contains(v->id());
         });
}

absl::Status VerifyOnlyPadRealToStaticReadsValue(
    const HloDataflowAnalysis& dataflow,
    const absl::flat_hash_set<HloValue::Id>& verified_value_ids,
    const HloValue& value, const HloValue& entry_parameter_value) {
  const HloInstruction* parameter =
      entry_parameter_value.defining_instruction();
  const ShapeIndex& leaf_index = entry_parameter_value.defining_index();
  auto format_error = [&](absl::string_view detail,
                          absl::string_view via_name) {
    return InvalidArgument(
        "Entry parameter %d%s of module %s feeds PadRealToStatic and is also "
        "%s (through %s). A dynamic parameter consumed by PadRealToStatic is "
        "passed as a compact buffer without dynamic-shape metadata, which no "
        "other instruction can read, so it must have no other users.",
        parameter->parameter_number(),
        leaf_index.empty() ? "" : absl::StrCat(" ", leaf_index.ToString()),
        parameter->GetModule()->name(), detail, via_name);
  };

  for (const HloPosition& pos : value.positions()) {
    if (pos.instruction->IsRoot()) {
      return format_error(absl::StrCat("returned as a result of computation ",
                                       pos.instruction->parent()->name()),
                          pos.instruction->name());
    }
  }
  for (const HloUse& use : value.GetUses()) {
    if (!IsAllowedPadRealToStaticUse(dataflow, verified_value_ids, value,
                                     use)) {
      const HloInstruction* via = use.instruction->operand(use.operand_number);
      return format_error(
          absl::StrCat("used by ", use.instruction->ToString(),
                       " in computation ", use.instruction->parent()->name()),
          via->name());
    }
  }
  return absl::OkStatus();
}

struct VerifiedPadValue {
  const HloValue* value;
  const HloValue* entry_origin;
};

struct PadRealToStaticResolver {
  const HloModule& module;
  const HloDataflowAnalysis& dataflow;
  absl::flat_hash_map<HloValue::Id, std::vector<const HloValue*>> memo = {};
  absl::flat_hash_set<HloValue::Id> active_stack = {};
  absl::flat_hash_set<HloValue::Id> verified_value_ids = {};
  std::vector<VerifiedPadValue> verified_values = {};

  absl::Status ResolvePadOperand(const HloInstruction* pad) {
    const std::vector<const HloValue*>& values =
        dataflow.GetValueSet(pad->operand(0), {}).values();
    if (values.empty()) {
      return InvalidArgument(
          "Dynamic operand of PadRealToStatic (%s) in module %s does not "
          "resolve to an entry computation parameter.",
          pad->ToString(), module.name());
    }
    for (const HloValue* value : values) {
      active_stack.clear();
      ABSL_RETURN_IF_ERROR(ResolveEntryOrigins(pad, value).status());
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<const HloValue*>> ResolveEntryOrigins(
      const HloInstruction* pad, const HloValue* value) {
    if (auto it = memo.find(value->id()); it != memo.end()) {
      return it->second;
    }

    const HloComputation* entry_comp = module.entry_computation();
    const HloInstruction* def = value->defining_instruction();
    auto non_entry_error = [&]() {
      return InvalidArgument(
          "Dynamic operand of PadRealToStatic (%s) in module %s must come from "
          "an entry computation parameter without being copied or recomputed, "
          "but its value is defined by %s in computation %s.",
          pad->ToString(), module.name(), def->ToString(),
          def->parent()->name());
    };

    if (def->opcode() != HloOpcode::kParameter) {
      return non_entry_error();
    }
    if (def->parent() == entry_comp) {
      const Shape& leaf_shape =
          ShapeUtil::GetSubshape(def->shape(), value->defining_index());
      if (leaf_shape.IsArray() && leaf_shape.is_dynamic() &&
          verified_value_ids.insert(value->id()).second) {
        verified_values.push_back({value, value});
      }
      std::vector<const HloValue*> result = {value};
      memo.emplace(value->id(), result);
      return result;
    }

    const HloComputation* sub_comp = def->parent();
    absl::InlinedVector<HloInstruction*, 1> callers =
        sub_comp->caller_instructions();
    if (callers.empty() || !active_stack.insert(value->id()).second) {
      return non_entry_error();
    }
    absl::c_sort(callers, [](const HloInstruction* a, const HloInstruction* b) {
      return a->unique_id() < b->unique_id();
    });

    std::vector<const HloValue*> resolved_entry_values;
    absl::flat_hash_set<HloValue::Id> seen_local_entry_values;
    for (const HloInstruction* caller : callers) {
      // `kAsyncStart` already covers the bound operands of its updates and
      // done.
      if ((caller->opcode() == HloOpcode::kAsyncUpdate ||
           caller->opcode() == HloOpcode::kAsyncDone) &&
          caller->async_chain_start() != nullptr &&
          caller->async_wrapped_computation() == sub_comp) {
        continue;
      }
      const HloInstruction* arg = GetSubcomputationCallerOperand(
          caller, sub_comp, def->parameter_number());
      if (arg == nullptr ||
          !ShapeUtil::IndexIsValid(arg->shape(), value->defining_index())) {
        active_stack.erase(value->id());
        return non_entry_error();
      }
      const std::vector<const HloValue*>& caller_values =
          dataflow.GetValueSet(arg, value->defining_index()).values();
      if (caller_values.empty()) {
        active_stack.erase(value->id());
        return non_entry_error();
      }
      for (const HloValue* caller_value : caller_values) {
        absl::StatusOr<std::vector<const HloValue*>> entry_vals =
            ResolveEntryOrigins(pad, caller_value);
        if (!entry_vals.ok()) {
          active_stack.erase(value->id());
          return entry_vals.status();
        }
        for (const HloValue* entry_val : *entry_vals) {
          if (seen_local_entry_values.insert(entry_val->id()).second) {
            resolved_entry_values.push_back(entry_val);
          }
        }
      }
    }
    active_stack.erase(value->id());

    if (resolved_entry_values.empty()) {
      return non_entry_error();
    }
    for (const HloValue* entry_val : resolved_entry_values) {
      const Shape& leaf_shape =
          ShapeUtil::GetSubshape(entry_val->defining_instruction()->shape(),
                                 entry_val->defining_index());
      if (leaf_shape.IsArray() && leaf_shape.is_dynamic()) {
        if (verified_value_ids.insert(value->id()).second) {
          verified_values.push_back({value, entry_val});
        }
        break;
      }
    }
    memo.emplace(value->id(), resolved_entry_values);
    return resolved_entry_values;
  }
};

}  // namespace

absl::Status ClearPadRealToStaticParameterMetadataPrefixes(
    const HloModule& hlo_module, absl::Span<Shape> parameter_shapes) {
  // Scan every computation (offloaded pads live outside the entry computation)
  // before running dataflow so modules without dynamic pads pay no cost.
  std::vector<const HloInstruction*> pad_calls;
  for (const HloComputation* comp : hlo_module.computations()) {
    for (const HloInstruction* instruction : comp->instructions()) {
      if (instruction->IsCustomCall("PadRealToStatic") &&
          instruction->operand_count() > 0 &&
          instruction->operand(0)->shape().IsArray() &&
          instruction->operand(0)->shape().is_dynamic()) {
        pad_calls.push_back(instruction);
      }
    }
  }
  if (pad_calls.empty()) {
    return absl::OkStatus();
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloDataflowAnalysis> dataflow,
                        HloDataflowAnalysis::Run(hlo_module));

  const HloComputation* entry_comp = hlo_module.entry_computation();
  const ComputationLayout& layout = hlo_module.entry_computation_layout();
  const bool flattened_tuple_parameter =
      layout.parameter_count() == 1 && layout.parameter_shape(0).IsTuple();

  // Every pad operand must trace to a zero-prefix entry parameter (directly via
  // dataflow or across embedded offload dispatchers); other defs carry +1024B.
  PadRealToStaticResolver resolver{hlo_module, *dataflow};
  for (const HloInstruction* pad : pad_calls) {
    ABSL_RETURN_IF_ERROR(resolver.ResolvePadOperand(pad));
  }

  std::vector<const HloValue*> parameter_values;
  parameter_values.reserve(resolver.verified_values.size());
  for (const VerifiedPadValue& item : resolver.verified_values) {
    const HloInstruction* entry_def = item.entry_origin->defining_instruction();
    CHECK_EQ(entry_def->opcode(), HloOpcode::kParameter);
    CHECK_EQ(entry_def->parent(), entry_comp);
    ABSL_RETURN_IF_ERROR(VerifyOnlyPadRealToStaticReadsValue(
        *dataflow, resolver.verified_value_ids, *item.value,
        *item.entry_origin));
    if (item.value->defining_instruction()->parent() != entry_comp) {
      continue;
    }
    if (std::optional<ShapeIndex> aliased_output =
            hlo_module.input_output_alias_config().GetAliasedOutput(
                entry_def->parameter_number(), item.value->defining_index());
        aliased_output.has_value()) {
      return InvalidArgument(
          "Entry parameter %d%s of module %s feeds PadRealToStatic and is "
          "aliased to output %s. A dynamic parameter consumed by "
          "PadRealToStatic is passed as a compact runtime-sized buffer without "
          "dynamic-shape metadata, so it cannot back an aliased or donated "
          "output.",
          entry_def->parameter_number(),
          item.value->defining_index().empty()
              ? ""
              : absl::StrCat(" ", item.value->defining_index().ToString()),
          hlo_module.name(), aliased_output->ToString());
    }
    parameter_values.push_back(item.value);
  }

  // Mutate `parameter_shapes` only after all validations succeed so an error
  // leaves `parameter_shapes` untouched.
  for (const HloValue* value : parameter_values) {
    int64_t shape_no = value->defining_instruction()->parameter_number();
    ShapeIndex index = value->defining_index();
    if (flattened_tuple_parameter) {
      if (shape_no != 0 || index.empty()) {
        continue;
      }
      shape_no = index.front();
      index = ShapeIndex(index.begin() + 1, index.end());
    }
    if (shape_no < 0 || shape_no >= parameter_shapes.size() ||
        !ShapeUtil::IndexIsValid(parameter_shapes[shape_no], index)) {
      continue;
    }
    Shape* subshape =
        ShapeUtil::GetMutableSubshape(&parameter_shapes[shape_no], index);
    if (subshape->IsArray() && subshape->has_layout()) {
      subshape->mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<PjRtLoadedExecutableDispatchInfo> InferDispatchInfo(
    const PjRtTopologyDescription* topology,
    std::vector<Shape> parameter_device_shapes, Shape output_device_shape,
    const HloInputOutputAliasConfig& alias_config,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices,
    std::unique_ptr<PjRtLoadedExecutableDispatchInfo::Extras> extras,
    bool tuple_inputs,
    std::unique_ptr<PjRtLoadedExecutableDispatchInfo::InputHloSnapshotBits>
        input_hlo_snapshot_bits) {
  PjRtLoadedExecutableDispatchInfo result{
      .parameter_device_shapes = std::move(parameter_device_shapes),
      .output_device_shape =
          std::make_shared<const Shape>(std::move(output_device_shape)),
      .addressable_devices = std::move(addressable_devices),
      .addressable_device_logical_ids =
          std::move(addressable_device_logical_ids),
      .device_assignment = std::move(device_assignment),
      .input_hlo_snapshot_bits = std::move(input_hlo_snapshot_bits),
      .extras = std::move(extras),
  };
  for (const auto& shape : result.parameter_device_shapes) {
    ABSL_ASSIGN_OR_RETURN(int kind,
                          topology->GetMemorySpaceKindForShape(shape));
    result.parameter_memory_space_kind_ids.push_back(kind);
  }
  {
    absl::Span<const Shape> shapes =
        result.output_device_shape->IsTuple()
            ? absl::MakeSpan(result.output_device_shape->tuple_shapes())
            : absl::MakeSpan(&*result.output_device_shape, 1);
    result.output_memory_space_kind_ids.reserve(shapes.size());
    for (const auto& shape : shapes) {
      ABSL_ASSIGN_OR_RETURN(int kind,
                            topology->GetMemorySpaceKindForShape(shape));
      result.output_memory_space_kind_ids.push_back(kind);
    }
  }
  // Initializes information about which arguments to which executables may
  // be donated due to aliases that were specified by the computation.
  ABSL_ASSIGN_OR_RETURN(
      result.parameters_that_may_be_donated,
      ComputeParametersThatMayBeDonated(
          alias_config, result.parameter_device_shapes.size(), tuple_inputs));
  result.input_buffer_sizes_in_bytes.reserve(
      result.parameter_device_shapes.size());
  for (const Shape& shape : result.parameter_device_shapes) {
    DCHECK(!shape.IsTuple());
    ABSL_ASSIGN_OR_RETURN(int64_t size_in_bytes,
                          PjRtGetOnDeviceBytesCount(shape));
    result.input_buffer_sizes_in_bytes.push_back(size_in_bytes);
  }
  return result;
}

namespace {

absl::StatusOr<std::vector<int64_t>> GetShardShape(
    const xla::HloSharding& sharding, llvm::ArrayRef<int64_t> shape,
    size_t num_devices) {
  if (sharding.IsReplicatedOrSingleDevice() || sharding.IsManual() ||
      sharding.IsUnreduced() || sharding.IsUnknown()) {
    return std::vector<int64_t>{shape.begin(), shape.end()};
  }
  if (sharding.TotalNumTiles() != num_devices) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "sharding's tile count and device count does not "
        "match: %d vs. %d; shape=%s, sharding=%s",
        sharding.TotalNumTiles(), num_devices, "??", sharding.ToString()));
  }
  if (shape.size() != sharding.TiledDataRank()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Numbers of dimensions don't match. From Shape %d vs from "
        "HloSharding %d",
        shape.size(), sharding.TiledDataRank()));
  }
  const absl::Span<const int64_t> sharding_dims = sharding.dimensions();
  std::vector<int64_t> tile_shape;
  tile_shape.reserve(shape.size());
  for (int64_t i = 0; i < shape.size(); ++i) {
    tile_shape.push_back(shape[i] == Shape::kUnboundedSize
                             ? Shape::kUnboundedSize
                             : xla::CeilOfRatio(shape[i], sharding_dims[i]));
  }
  return tile_shape;
}

}  // namespace

absl::StatusOr<PjRtLoadedExecutableDispatchInfo> InferDispatchInfo(
    const PjRtTopologyDescription* topology, mlir::ModuleOp mlir_module,
    const CompileOptions& options,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, bool tuple_inputs) {
  if (!device_assignment) {
    return absl::UnimplementedError(
        "Async compilation requires a device_assignment");
  }
  auto extras = std::make_unique<PjRtLoadedExecutableDispatchInfo::Extras>();
  extras->name =
      std::string(mlir_module.getSymName().value_or("?unknown program name?"));
  extras->num_partitions = device_assignment->replica_count();
  extras->num_replicas = device_assignment->replica_count();
  auto mesh = mlir_module.getOps<mlir::sdy::MeshOp>();
  if (mesh.empty() || std::next(mesh.begin()) != mesh.end()) {
    return absl::UnimplementedError("Exactly one shardy mesh is required.");
  }
  auto main = mlir_module.lookupSymbol<mlir::func::FuncOp>("main");
  if (main == nullptr) {
    return absl::UnimplementedError(
        "InferAvalFromModule called on module without main function.");
  }
  auto sharding_fn = [&](mlir::sdy::TensorShardingAttr sharding) {
    return sharding.getMesh(main);
  };
  auto get_xla_shape = [&](xla::HloSharding sharding, mlir::Type type)
      -> absl::StatusOr<
          std::tuple<xla::Shape, std::shared_ptr<const xla::PjRtLayout>>> {
    if (mlir::isa<mlir::UnrankedTensorType>(type)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported type ", mlir::debugString(type)));
    }
    std::vector<int64_t> shard_shape;
    xla::PrimitiveType primitive_type;
    std::vector<bool> dynamic_dimensions;
    if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
      xla::Shape unsharded_shape = xla::TypeToShape(tensor_type);
      if (unsharded_shape.element_type() == PRIMITIVE_TYPE_INVALID) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported type ", mlir::debugString(type)));
      }
      llvm::ArrayRef<int64_t> dims(unsharded_shape.dimensions().data(),
                                   unsharded_shape.dimensions().size());
      ABSL_ASSIGN_OR_RETURN(
          shard_shape,
          GetShardShape(sharding, dims,
                        extras->num_replicas * extras->num_partitions));
      primitive_type = unsharded_shape.element_type();
      dynamic_dimensions.assign(unsharded_shape.dynamic_dimensions().begin(),
                                unsharded_shape.dynamic_dimensions().end());
    } else {
      primitive_type = xla::ConvertMlirTypeToPrimitiveType(type);
    }
    ABSL_ASSIGN_OR_RETURN(auto* memory_space,
                          addressable_devices[0]->default_memory_space());
    xla::Shape xla_shard_shape = xla::ShapeUtil::MakeShape(
        primitive_type, shard_shape, dynamic_dimensions);
    // TODO(parkers): Fix the nullptr layout.
    ABSL_ASSIGN_OR_RETURN(
        auto xla_shape, topology->MakeCanonicalShapeForMemorySpace(
                            memory_space->kind_id(), xla_shard_shape, nullptr));
    auto layout = std::make_shared<PjRtLayout>(xla_shape.layout());
    return std::make_tuple(xla_shape, layout);
  };
  std::vector<xla::Shape> parameter_device_shapes;
  extras->parameter_shardings.emplace();
  extras->parameter_layouts = std::vector<std::shared_ptr<const PjRtLayout>>();
  for (size_t i = 0; i < main.getNumArguments(); ++i) {
    auto sharding_attr = mlir::sdy::getSharding(main.getArgument(i));
    if (sharding_attr == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Arg[%d] is missing a sharding", i));
    }
    xla::HloSharding hlo_sharding =
        xla::sdy::convertToHloSharding(sharding_attr, sharding_fn,
                                       /*manualAxes=*/{});
    xla::Shape shape;
    std::shared_ptr<const xla::PjRtLayout> layout;
    ABSL_ASSIGN_OR_RETURN(
        (std::tie(shape, layout)),
        get_xla_shape(hlo_sharding, main.getArgumentTypes()[i]));

    // `PadRealToStatic` inputs carry no metadata prefix. Async compilation
    // never reaches the HLO-based clearing, so mirror its checks here.
    if (shape.is_dynamic()) {
      mlir::BlockArgument arg = main.getArgument(i);
      // Operand 0 is the data tensor; the other operands are the sizes.
      const bool is_pad_real_to_static_input =
          llvm::any_of(arg.getUses(), [](mlir::OpOperand& use) {
            return use.getOperandNumber() == 0 &&
                   IsPadRealToStaticOp(use.getOwner());
          });
      if (is_pad_real_to_static_input) {
        if (!arg.hasOneUse()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Entry argument %d feeds PadRealToStatic and has other users; "
              "PadRealToStatic dynamic inputs have no metadata prefix and "
              "must have no other users.",
              i));
        }
        if (main.getArgAttr(i, "tf.aliasing_output") != nullptr) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Entry argument %d feeds PadRealToStatic and is aliased via "
              "tf.aliasing_output; compact PadRealToStatic buffers cannot back "
              "aliased or donated outputs.",
              i));
        }
        shape.mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
        layout = std::make_shared<PjRtLayout>(shape.layout());
      }
    }

    parameter_device_shapes.push_back(shape);
    extras->parameter_shardings->push_back(hlo_sharding.ToProto());
    extras->parameter_layouts->push_back(std::move(layout));
  }
  std::vector<xla::Shape> result_shapes;
  extras->output_shardings.emplace();
  extras->output_layouts = std::vector<std::shared_ptr<const PjRtLayout>>();
  for (size_t i = 0; i < main.getNumResults(); ++i) {
    auto sharding_attr = mlir::sdy::getFuncResultSharding(main, i);
    // sharding_attr.dump();
    if (sharding_attr == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Result[%d] is missing a sharding", i));
    }
    xla::HloSharding hlo_sharding =
        xla::sdy::convertToHloSharding(sharding_attr, sharding_fn,
                                       /*manualAxes=*/{});

    xla::Shape shape;
    std::shared_ptr<const xla::PjRtLayout> layout;
    ABSL_ASSIGN_OR_RETURN(
        (std::tie(shape, layout)),
        get_xla_shape(hlo_sharding, main.getResultTypes()[i]));

    result_shapes.push_back(std::move(shape));
    extras->output_shardings->push_back(hlo_sharding.ToProto());
    extras->output_layouts->push_back(std::move(layout));
  }
  Shape output_device_shape = (result_shapes.size() == 1)
                                  ? std::move(result_shapes[0])
                                  : xla::Shape(std::move(result_shapes));

  // TODO(parkers): implement aliasing.
  extras->input_output_alias_config =
      HloInputOutputAliasConfig(output_device_shape);

  const auto& input_output_alias_config = extras->input_output_alias_config;
  ABSL_ASSIGN_OR_RETURN(
      auto result,
      InferDispatchInfo(topology, std::move(parameter_device_shapes),
                        std::move(output_device_shape),
                        input_output_alias_config, std::move(device_assignment),
                        std::move(addressable_device_logical_ids),
                        std::move(addressable_devices), std::move(extras),
                        tuple_inputs));

  result.extras->fingerprint = "";
  result.extras->parameter_memory_kinds.reserve(
      result.parameter_memory_space_kind_ids.size());
  for (size_t i = 0; i < result.parameter_memory_space_kind_ids.size(); ++i) {
    auto* device = result.addressable_devices[0];
    ABSL_ASSIGN_OR_RETURN(auto* memory_space, device->default_memory_space());
    for (auto* ms : device->memory_spaces()) {
      if (ms->kind_id() == result.parameter_memory_space_kind_ids[i]) {
        memory_space = ms;
      }
    }
    result.extras->parameter_memory_kinds.push_back(memory_space->kind());
  }
  result.extras->output_memory_kinds.reserve(
      result.output_memory_space_kind_ids.size());
  for (size_t i = 0; i < result.output_memory_space_kind_ids.size(); ++i) {
    auto* device = result.addressable_devices[0];
    ABSL_ASSIGN_OR_RETURN(auto* memory_space, device->default_memory_space());
    for (auto* ms : device->memory_spaces()) {
      if (ms->kind_id() == result.output_memory_space_kind_ids[i]) {
        memory_space = ms;
      }
    }
    result.extras->output_memory_kinds.push_back(memory_space->kind());
  }
  return result;
}

}  // namespace xla
