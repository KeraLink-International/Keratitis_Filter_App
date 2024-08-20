import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import numpy as np
import av

st.title("Keratitis Vision Simulator")

# Custom CSS to hide the original select device button
hide_device_button_css = """
<style>
    button[title="Select device"] {
        display: none;
    }
</style>
"""

st.markdown(hide_device_button_css, unsafe_allow_html=True)

# Custom JavaScript to add a "Select Camera" button
select_camera_button_js = """
<script>
    const selectDeviceButton = document.querySelector('button[title="Select device"]');
    if (selectDeviceButton) {
        const customButton = document.createElement('button');
        customButton.innerHTML = 'Select Camera';
        customButton.onclick = () => selectDeviceButton.click();
        selectDeviceButton.parentNode.insertBefore(customButton, selectDeviceButton.nextSibling);
    }
</script>
"""

st.markdown(select_camera_button_js, unsafe_allow_html=True)

filter_options = ["Healthy Eye", "Early Stage (<1 Week)", "Middle Stage (2 Weeks)", "Late Stage (>3 Weeks)"]
filter = st.selectbox("Select Severity", filter_options, index=0)

line_position = st.slider("Adjust Filter Position", min_value=0, max_value=100, value=50, step=1)

st.logo("https://www.iapb.org/wp-content/uploads/2020/09/KeraLink-International.png", link="https://www.keralink.org/")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.noise_pattern = None

    def generate_fixed_noise_pattern(self, height, width):
        base_tint_color = 128
        noise_intensity = 3
        random_noise = np.random.randint(-noise_intensity, noise_intensity, (height, width, 3), dtype=np.int16)
        noise_pattern = np.clip(base_tint_color + random_noise, 0, 255).astype(np.uint8)
        return noise_pattern

    def apply_filter_to_area(self, img, filter_type):
        height, width, _ = img.shape

        noise_pattern = self.generate_fixed_noise_pattern(height, width)

        params = {
            "Early Stage (<1 Week)": {"opacity": 0.4, "blur_radius": 31, "outer_blur_radius": 21},
            "Middle Stage (2 Weeks)": {"opacity": 0.2, "blur_radius": 51, "outer_blur_radius": 41},
            "Late Stage (>3 Weeks)": {"opacity": 0.1, "blur_radius": 91, "outer_blur_radius": 81}
        }

        p = params.get(filter_type)

        blurred_img = cv2.GaussianBlur(img, (p["blur_radius"], p["outer_blur_radius"]), 0)

        tinted_blurred_img = cv2.addWeighted(blurred_img, p["opacity"], noise_pattern, 1 - p["opacity"], 0)

        return tinted_blurred_img

    def transform(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        height, width, _ = img.shape

        split_point = int((line_position / 100) * width)
        left_half = img[:, :split_point]
        right_half = img[:, split_point:]

        if filter != "Healthy Eye":
            right_half = self.apply_filter_to_area(right_half, filter)

        img = np.concatenate((left_half, right_half), axis=1)
        cv2.line(img, (split_point, 0), (split_point, height), (255, 255, 255), 3)

        video_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        
        del img

        return video_frame

rtc_config = RTCConfiguration({
    "iceServers": [{"urls": "stun:stun.l.google.com:19302"}]
})

webrtc_streamer(
    key="streamer",
    video_frame_callback=VideoProcessor().transform,
    rtc_configuration=rtc_config,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 30}},
        "audio": False,
    }
)
