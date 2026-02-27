"""RTSP to MJPEG stream proxy for dashboard camera view"""

import asyncio
import logging
import time

from aiohttp import web

logger = logging.getLogger(__name__)


async def camera_stream(request: web.Request) -> web.StreamResponse:
    """Proxy RTSP camera feed as MJPEG multipart stream"""
    app_state = request.app["app_state"]
    rtsp_url = app_state.get("camera_rtsp_url")
    fps = app_state.get("camera_mjpeg_fps", 5)
    quality = app_state.get("camera_mjpeg_quality", 80)

    if not rtsp_url:
        raise web.HTTPServiceUnavailable(text="Camera not configured")

    # Lazy import opencv to avoid startup cost if camera unused
    try:
        import cv2
    except ImportError:
        raise web.HTTPServiceUnavailable(text="OpenCV not available")

    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )
    await response.prepare(request)

    cap = None
    frame_interval = 1.0 / fps

    try:
        cap = await asyncio.to_thread(cv2.VideoCapture, rtsp_url)
        if not cap.isOpened():
            logger.error(f"Failed to open RTSP stream: {rtsp_url}")
            return response

        logger.info(f"Camera stream started for client {request.remote}")

        while not response.task.done():
            start = time.monotonic()

            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                logger.warning("Failed to read camera frame, reconnecting...")
                await asyncio.sleep(1.0)
                cap.release()
                cap = await asyncio.to_thread(cv2.VideoCapture, rtsp_url)
                continue

            # Encode as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            ret, jpeg = await asyncio.to_thread(cv2.imencode, ".jpg", frame, encode_params)
            if not ret:
                continue

            jpeg_bytes = jpeg.tobytes()

            try:
                await response.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n"
                    b"\r\n" + jpeg_bytes + b"\r\n"
                )
            except (ConnectionResetError, ConnectionAbortedError):
                break

            # Throttle to target FPS
            elapsed = time.monotonic() - start
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Camera stream error: {e}")
    finally:
        if cap is not None:
            cap.release()
        logger.info(f"Camera stream ended for client {request.remote}")

    return response
