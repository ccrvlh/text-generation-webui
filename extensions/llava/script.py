import gradio as gr

from app.utils.logging import logger


def ui():
    gr.Markdown("### This extension is deprecated, use \"multimodal\" extension instead")
    logger.error("LLaVA extension is deprecated, use \"multimodal\" extension instead")
