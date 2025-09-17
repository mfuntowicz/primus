// Copyright Morgan Funtowicz (c) 2025. 
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================*

//
// Created by momo- on 9/15/2025.
//

#ifndef UNICRON_ERRORS_H
#define UNICRON_ERRORS_H

#include <string>

namespace unicron {
    enum error_kind_t : int32_t {
        HostTargetDetectionFailed = 0,
        ModuleAlreadyExist = 1,
        ModuleCompilationFailed = 2
    };

    struct error_t {
        const error_kind_t kind;
        const std::string reason;

        static error_t host_target_detection_failed(const std::string &msg) {
            return error_t{HostTargetDetectionFailed, msg};
        }

        static error_t module_already_exist(const std::string &msg) {
            return error_t{ModuleAlreadyExist, msg};
        }

        static error_t module_compilation_failed(const std::string &msg) {
            return error_t{ModuleCompilationFailed, msg};
        }
    };
}

#endif //UNICRON_ERRORS_H