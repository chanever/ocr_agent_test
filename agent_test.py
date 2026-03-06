import base64
import os
from playwright.sync_api import sync_playwright


def take_screenshot():

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=False,
            args=["--start-maximized"]
        )

        page = browser.new_page(no_viewport=True)

        # 테스트 사이트
        page.goto("https://naver.com")

        # screenshot 저장
        screenshot_path = "screen.png"
        # DOM 로딩만 기다림
        page.wait_for_load_state("domcontentloaded")

        # 추가 로딩 대기
        page.wait_for_timeout(2000)
        
        page.screenshot(path=screenshot_path)

        print("✅ Screenshot saved:", screenshot_path)

        browser.close()

        return screenshot_path


def load_image(image_path):

    if not os.path.exists(image_path):
        print("❌ 이미지 파일 없음")
        return None

    print("✅ 이미지 파일 존재")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print("✅ 이미지 파일 읽기 성공")
    print("이미지 크기(bytes):", len(image_bytes))

    return image_bytes


def encode_base64(image_bytes):

    base64_image = base64.b64encode(image_bytes).decode()

    print("✅ Base64 인코딩 성공")

    print("Base64 길이:", len(base64_image))

    # 앞부분 일부만 출력
    print("Base64 preview:", base64_image[:100])

    return base64_image


def main():

    print("\n===== STEP 1: Screenshot =====")

    image_path = take_screenshot()

    print("\n===== STEP 2: Load Image =====")

    image_bytes = load_image(image_path)

    print("\n===== STEP 3: Base64 Encoding =====")

    base64_image = encode_base64(image_bytes)

    print("\n🎉 테스트 완료")
    print("이제 GPT Vision API에 넣을 준비 완료")


if __name__ == "__main__":
    main()