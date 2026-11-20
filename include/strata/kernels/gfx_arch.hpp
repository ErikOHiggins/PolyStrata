// include/strata/kernels/gfx_arch.hpp - AMD architecture names as HIP reports them (hipDeviceProp_t::gcnArchName).
// Pure string matching (no HIP headers), usable from host code of any build.
#pragma once

#include <cstddef>
#include <cstring>

namespace strata::kernels {

/// `gcn` is hipDeviceProp_t::gcnArchName ("gfx1151", or "gfx1151:sramecc-:xnack-"): true when it is exactly `name`.
inline bool gfx_arch_is(const char* gcn, const char* name) {
    if (gcn == nullptr) return false;
    const std::size_t n = std::strlen(name);
    return std::strncmp(gcn, name, n) == 0 && (gcn[n] == '\0' || gcn[n] == ':');
}

/// RDNA4 (Radeon AI PRO R9700 = gfx1201, RX 9070 / 9070 XT = gfx1201, RX 9060 = gfx1200).
inline bool gfx_arch_is_gfx12(const char* gcn) { return gfx_arch_is(gcn, "gfx1200") || gfx_arch_is(gcn, "gfx1201"); }

}  // namespace strata::kernels
