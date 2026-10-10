/***************************************************************************
*  Copyright 2020-2022 Intel Corporation
*  Copyright (C) Codeplay Software Limited
*  Modified 2026 by MistVVK and the XeStrata contributors: the device made current with hipSetDevice and
*  the handles kept per device, in place of HIP's deprecated context API.
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/
#include "rocblas_scope_handle.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace rocblas {

template <typename T>
rocblas_handle_container<T>::~rocblas_handle_container() noexcept(false) {
    for (auto& handle_pair : rocblas_handle_container_mapper_) {
        rocblas_status err;
        if (handle_pair.second != nullptr) {
            auto handle = handle_pair.second->exchange(nullptr);
            if (handle != nullptr) {
                ROCBLAS_ERROR_FUNC(rocblas_destroy_handle, err, handle);
                handle = nullptr;
            }
            else {
                delete handle_pair.second;
            }
            handle_pair.second = nullptr;
        }
    }
    rocblas_handle_container_mapper_.clear();
}

/**
 * Inserts a new element in the map if its key is unique. This new element
 * is constructed in place using args as the arguments for the construction
 * of a value_type (which is an object of a pair type). The insertion only
 * takes place if no other element in the container has a key equivalent to
 * the one being emplaced (keys in a map container are unique).
 */
thread_local rocblas_handle_container<int> RocblasScopedContextHandler::handle_helper =
    rocblas_handle_container<int>{};

RocblasScopedContextHandler::RocblasScopedContextHandler(sycl::queue queue,
                                                         sycl::interop_handle& ih)
        : interop_h(ih),
          needToRecover_(false) {
    placedContext_ = new sycl::context(queue.get_context());
    int hipDevice = ih.get_native_device<sycl::backend::ext_oneapi_hip>();
    hipError_t err;
    // The device made current selects its primary context (HIP's context API is deprecated); the
    // previous one comes back in the destructor
    HIP_ERROR_FUNC(hipGetDevice, err, &original_);
    if (original_ != hipDevice) {
        HIP_ERROR_FUNC(hipSetDevice, err, hipDevice);
        needToRecover_ = true;
    }
}

RocblasScopedContextHandler::~RocblasScopedContextHandler() noexcept(false) {
    if (needToRecover_) {
        hipError_t err;
        HIP_ERROR_FUNC(hipSetDevice, err, original_);
    }
    delete placedContext_;
}

void ContextCallback(void* userData) {
    auto* ptr = static_cast<std::atomic<rocblas_handle>*>(userData);
    if (!ptr) {
        return;
    }
    auto handle = ptr->exchange(nullptr);
    if (handle != nullptr) {
        rocblas_status err1;
        ROCBLAS_ERROR_FUNC(rocblas_destroy_handle, err1, handle);
        handle = nullptr;
    }
    else {
        // if the handle is nullptr it means the handle was already destroyed by
        // the rocblas_handle destructor and we're free to delete the atomic
        // object.
        delete ptr;
    }
}

rocblas_handle RocblasScopedContextHandler::get_handle(const sycl::queue& queue) {
    int hipDevice = interop_h.get_native_device<sycl::backend::ext_oneapi_hip>();
    hipStream_t streamId = get_stream(queue);
    rocblas_status err;
    auto it = handle_helper.rocblas_handle_container_mapper_.find(hipDevice);
    if (it != handle_helper.rocblas_handle_container_mapper_.end()) {
        if (it->second == nullptr) {
            handle_helper.rocblas_handle_container_mapper_.erase(it);
        }
        else {
            auto handle = it->second->load();
            if (handle != nullptr) {
                hipStream_t currentStreamId;
                ROCBLAS_ERROR_FUNC(rocblas_get_stream, err, handle, &currentStreamId);
                if (currentStreamId != streamId) {
                    ROCBLAS_ERROR_FUNC(rocblas_set_stream, err, handle, streamId);
                }
                return handle;
            }
            else {
                handle_helper.rocblas_handle_container_mapper_.erase(it);
            }
        }
    }

    rocblas_handle handle;

    ROCBLAS_ERROR_FUNC(rocblas_create_handle, err, &handle);
    ROCBLAS_ERROR_FUNC(rocblas_set_stream, err, handle, streamId);

    auto insert_iter = handle_helper.rocblas_handle_container_mapper_.insert(
        std::make_pair(hipDevice, new std::atomic<rocblas_handle>(handle)));

    sycl::detail::pi::contextSetExtendedDeleter(*placedContext_, ContextCallback,
                                                insert_iter.first->second);
    return handle;
}

hipStream_t RocblasScopedContextHandler::get_stream(const sycl::queue& queue) {
    // The stream must come from the interop handle rather than from the queue:
    // a queue owns a pool of streams and only the one the command was scheduled
    // on is tracked by the SYCL event that gates this command's completion.
    return interop_h.get_native_queue<sycl::backend::ext_oneapi_hip>();
}
sycl::context RocblasScopedContextHandler::get_context(const sycl::queue& queue) {
    return queue.get_context();
}

} // namespace rocblas
} // namespace blas
} // namespace math
} // namespace oneapi
