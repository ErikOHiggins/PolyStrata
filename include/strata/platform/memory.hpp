// include/strata/platform/memory.hpp - the GPU's memory budget as Windows reports it.
#pragma once

#include <cstdint>
#include <string>

namespace strata::platform {

/// Windows: the budget Windows gives this process in the GPU's local (video) memory, which falls as the process grows
/// and as other programs take their share, and what the process uses of it, from DXGI
/// (IDXGIAdapter3::QueryVideoMemoryInfo, DXGI_MEMORY_SEGMENT_GROUP_LOCAL) for the adapter whose LUID is the 8 bytes at
/// `luid` (cudaDeviceProp::luid).  Memory past the budget is moved out to system RAM (WDDM), and a card that does that
/// runs at a fraction of its speed.  False (and `why` says so) when the query is not possible - always elsewhere than
/// Windows.
bool gpu_local_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why);

}  // namespace strata::platform
