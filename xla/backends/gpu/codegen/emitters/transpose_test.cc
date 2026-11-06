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
#include "xla/backends/gpu/codegen/emitters/transpose.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

using TransposeFusionTest = HloHardwareIndependentTestBase;

TEST_F(TransposeFusionTest,
       PackedTransposeUses256ThreadsForSubByteAnd8BitTypesOnCuda) {
  se::DeviceDescription device_info = TestGpuDeviceInfo::B200SXMDeviceInfo();

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> s4_module,
                       ParseAndReturnVerifiedModule(R"hlo(
    HloModule s4_packed_transpose
    fusion {
      p0 = s4[2,4096,16384]{2,1,0:E(4)} parameter(0)
      ROOT t = s4[2,16384,4096]{2,1,0:E(4)} transpose(p0), dimensions={0,2,1}
    }
    ENTRY main {
      p0 = s4[2,4096,16384]{2,1,0:E(4)} parameter(0)
      ROOT root = s4[2,16384,4096]{2,1,0:E(4)} fusion(p0), kind=kInput, calls=fusion
    }
  )hlo"));
  HloFusionAnalysis s4_analysis = HloFusionAnalysis::Create(
      *s4_module->entry_computation()->root_instruction(), device_info);
  std::unique_ptr<MlirKernelEmitter> s4_emitter =
      CreateTransposeFusion(s4_analysis);
  ASSERT_NE(dynamic_cast<const PackedTranspose*>(s4_emitter.get()), nullptr);
  EXPECT_EQ(s4_emitter->launch_dimensions().num_threads_per_block(), 256);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> s8_module,
                       ParseAndReturnVerifiedModule(R"hlo(
    HloModule s8_packed_transpose
    fusion {
      p0 = s8[2,4096,8192]{2,1,0} parameter(0)
      ROOT t = s8[2,8192,4096]{2,1,0} transpose(p0), dimensions={0,2,1}
    }
    ENTRY main {
      p0 = s8[2,4096,8192]{2,1,0} parameter(0)
      ROOT root = s8[2,8192,4096]{2,1,0} fusion(p0), kind=kInput, calls=fusion
    }
  )hlo"));
  HloFusionAnalysis s8_analysis = HloFusionAnalysis::Create(
      *s8_module->entry_computation()->root_instruction(), device_info);
  std::unique_ptr<MlirKernelEmitter> s8_emitter =
      CreateTransposeFusion(s8_analysis);
  ASSERT_NE(dynamic_cast<const PackedTranspose*>(s8_emitter.get()), nullptr);
  EXPECT_EQ(s8_emitter->launch_dimensions().num_threads_per_block(), 256);
}

TEST_F(TransposeFusionTest, PackedTransposeUses128ThreadsFor16BitTypesOnCuda) {
  se::DeviceDescription device_info = TestGpuDeviceInfo::B200SXMDeviceInfo();

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> bf16_module,
                       ParseAndReturnVerifiedModule(R"hlo(
    HloModule bf16_packed_transpose
    fusion {
      p0 = bf16[2048,16,32]{2,1,0} parameter(0)
      ROOT t = bf16[32,16,2048]{2,1,0} transpose(p0), dimensions={2,1,0}
    }
    ENTRY main {
      p0 = bf16[2048,16,32]{2,1,0} parameter(0)
      ROOT root = bf16[32,16,2048]{2,1,0} fusion(p0), kind=kInput, calls=fusion
    }
  )hlo"));
  HloFusionAnalysis bf16_analysis = HloFusionAnalysis::Create(
      *bf16_module->entry_computation()->root_instruction(), device_info);
  std::unique_ptr<MlirKernelEmitter> bf16_emitter =
      CreateTransposeFusion(bf16_analysis);
  ASSERT_NE(dynamic_cast<const PackedTranspose*>(bf16_emitter.get()), nullptr);
  EXPECT_EQ(bf16_emitter->launch_dimensions().num_threads_per_block(), 128);
}

TEST_F(TransposeFusionTest, PackedTransposeUses256ThreadsFor16BitTypesOnRocm) {
  se::DeviceDescription device_info = TestGpuDeviceInfo::AMDMI210DeviceInfo();

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> bf16_module,
                       ParseAndReturnVerifiedModule(R"hlo(
    HloModule bf16_rocm_packed_transpose
    fusion {
      p0 = bf16[2048,16,32]{2,1,0} parameter(0)
      ROOT t = bf16[32,16,2048]{2,1,0} transpose(p0), dimensions={2,1,0}
    }
    ENTRY main {
      p0 = bf16[2048,16,32]{2,1,0} parameter(0)
      ROOT root = bf16[32,16,2048]{2,1,0} fusion(p0), kind=kInput, calls=fusion
    }
  )hlo"));
  HloFusionAnalysis bf16_analysis = HloFusionAnalysis::Create(
      *bf16_module->entry_computation()->root_instruction(), device_info);
  std::unique_ptr<MlirKernelEmitter> bf16_emitter =
      CreateTransposeFusion(bf16_analysis);
  ASSERT_NE(dynamic_cast<const PackedTranspose*>(bf16_emitter.get()), nullptr);
  EXPECT_EQ(bf16_emitter->launch_dimensions().num_threads_per_block(), 256);
}

}  // namespace
}  // namespace xla::gpu
