import streamlit as st
from PIL import Image
import io
import torch
from prompt_advisor import render_prompt_advisor

st.set_page_config(page_title="Imagen AI", page_icon="🎨", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f0f0f; }
    [data-testid="stSidebar"] { background: #1a1a1a; border-right: 1px solid #2a2a2a; }
    .title {
        font-size: 2.6rem; font-weight: 900; text-align: center;
        background: linear-gradient(90deg, #f97316, #ef4444, #a855f7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 0.5rem 0 0.2rem 0;
    }
    .section-label {
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em;
        color: #6b7280; text-transform: uppercase; margin-bottom: 0.3rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #f97316, #ef4444) !important;
        color: white !important; font-weight: 800 !important;
        font-size: 1rem !important; border: none !important;
        border-radius: 10px !important; padding: 0.65rem !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎨 Imagen AI</div>', unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#6b7280;font-size:0.9rem;margin-bottom:1.5rem'>Powered by Juggernaut XL · AI Prompt Advisor</div>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    mode = st.radio("Generation Mode", ["⚡ Fast (LCM ~30s)", "🎨 Quality (Normal ~2min)"])
    fast_mode = "Fast" in mode

    if fast_mode:
        steps    = st.slider("Steps", 4, 12, 6)
        guidance = st.slider("Guidance Scale", 1.0, 3.0, 1.5, step=0.5)
        st.info("⚡ LCM: 6 steps ≈ 30 seconds")
    else:
        steps    = st.slider("Steps", 10, 50, 20)
        guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.0, step=0.5)

    st.markdown("---")
    width  = st.select_slider("Width",  options=[512, 640, 768], value=512)
    height = st.select_slider("Height", options=[512, 640, 768], value=512)
    seed   = st.number_input("Seed (-1 = random)", min_value=-1, max_value=999999, value=-1)
    st.markdown("---")
    device_info = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🟡 CPU (slow)"
    st.info(f"Device: {device_info}")

# ── Layout: Left | Divider | Right ───────────────────────────────────────────
left_col, mid_col, right_col = st.columns([10, 0.1, 10], gap="small")

# ════════════════════════════════════════
# LEFT — Image Generation
# ════════════════════════════════════════
with left_col:
    st.markdown("#### 🖼️ Image Generation")
    st.markdown("<div style='height:3px;background:linear-gradient(90deg,#f97316,#ef4444);border-radius:4px;margin-bottom:1rem'></div>", unsafe_allow_html=True)

    st.markdown('<p class="section-label">Prompt</p>', unsafe_allow_html=True)
    prompt = st.text_area("", height=110, label_visibility="collapsed",
        placeholder="RAW photo, a beautiful mountain landscape, golden hour, photorealistic, 8k, sharp focus",
        key="main_prompt")

    st.markdown('<p class="section-label">Negative Prompt (Recommended)</p>', unsafe_allow_html=True)
    neg_prompt = st.text_area("", height=70, label_visibility="collapsed",
        value="worst quality, low quality, blurry, watermark, ugly, deformed, bad anatomy, bad hands, extra fingers, cropped, jpeg artifacts",
        key="neg_prompt")

    st.markdown('<p class="section-label">Reference Image (optional)</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    img2img_strength = 0.6
    if uploaded:
        st.image(uploaded, use_container_width=True)
        img2img_strength = st.slider("Denoising Strength", 0.1, 1.0, 0.6, step=0.05)

    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
    generate = st.button("⚡ Generate Image")

    @st.cache_resource(show_spinner=False)
    def load_pipeline(fast_mode, has_image):
        from diffusers import (StableDiffusionXLPipeline,
                               StableDiffusionXLImg2ImgPipeline,
                               LCMScheduler)
        model_id = "RunDiffusion/Juggernaut-XL-v9"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if has_image:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=dtype, use_safetensors=True, variant="fp16")
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=dtype, use_safetensors=True, variant="fp16")

        if fast_mode:
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
            pipe.fuse_lora()
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        pipe.enable_attention_slicing(1)
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()
        return pipe

    if generate:
        if not prompt.strip():
            st.error("Please enter a prompt.")
        else:
            has_image = uploaded is not None
            with st.spinner("⏳ Loading model..."):
                try:
                    pipe = load_pipeline(fast_mode, has_image)
                except Exception as e:
                    st.error(f"Model load error: {str(e)}")
                    st.stop()

            generator = None if seed == -1 else torch.Generator("cuda").manual_seed(int(seed))
            mode_label = "⚡ LCM fast" if fast_mode else "🎨 quality"

            with st.spinner(f"🎨 Generating ({mode_label}, {steps} steps)..."):
                try:
                    if has_image:
                        img_bytes = uploaded.read()
                        ref_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((width, height))
                        result    = pipe(
                            prompt=prompt, negative_prompt=neg_prompt,
                            image=ref_img, strength=img2img_strength,
                            num_inference_steps=steps, guidance_scale=guidance,
                            generator=generator,
                        ).images[0]
                    else:
                        result = pipe(
                            prompt=prompt, negative_prompt=neg_prompt,
                            width=width, height=height,
                            num_inference_steps=steps, guidance_scale=guidance,
                            generator=generator,
                        ).images[0]

                    st.success("✅ Image generated!")
                    st.image(result, use_container_width=True)

                    dl_buf = io.BytesIO()
                    result.save(dl_buf, format="PNG")
                    dl_buf.seek(0)
                    st.download_button("⬇️ Download Image", data=dl_buf,
                                       file_name="juggernaut_output.png", mime="image/png")

                except Exception as e:
                    err = str(e)
                    st.error(f"Generation error: {err}")
                    if "out of memory" in err.lower():
                        st.info("💡 OOM — reduce to 512x512 and 6 steps.")

# ════════════════════════════════════════
# MIDDLE — Divider line
# ════════════════════════════════════════
with mid_col:
    st.markdown("<div style='border-left:1px solid #2a2a2a;min-height:700px;margin:0 auto;'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════
# RIGHT — AI Prompt Advisor
# ════════════════════════════════════════
with right_col:
    st.markdown("#### 🤖 AI Prompt Advisor")
    st.markdown("<div style='height:3px;background:linear-gradient(90deg,#a855f7,#3b82f6);border-radius:4px;margin-bottom:1rem'></div>", unsafe_allow_html=True)
    render_prompt_advisor()