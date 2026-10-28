// Shared by Metal host and shaders only. CUDA retains its upstream variants.
#pragma once
#if defined(NH_MSG_V4096)
#error "Metal NetHack does not support NH_MSG_V4096; use the default 1024-bucket encoder"
#endif
#if defined(NH_MSG128)
#error "Metal NetHack does not support NH_MSG128; use the default 256-wide message encoder"
#endif
#if defined(NH_NO_APANEL)
#error "Metal NetHack does not support NH_NO_APANEL; the default accessory panel is required"
#endif
#if defined(NETHACK_V3_K) && NETHACK_V3_K != 16
#error "Metal NetHack requires NETHACK_V3_K=16"
#endif
