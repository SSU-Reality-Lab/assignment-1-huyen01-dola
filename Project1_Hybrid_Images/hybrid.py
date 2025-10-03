import sys
import cv2
import numpy as np
import os

def gaussian_blur_kernel_2d(sigma, height, width):
    '''
    2D Gaussian kernel 생성 (height x width 크기).
    sigma에 따라 퍼짐 정도 결정.
    '''
    # 중심 좌표 계산
    y_mid = height // 2
    x_mid = width // 2

    kernel = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            dy = y - y_mid
            dx = x - x_mid
            kernel[y, x] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    # Normalize (합이 1이 되도록)
    kernel /= np.sum(kernel)
    return kernel


def cross_correlation_2d(img, kernel):
    '''
    2D cross-correlation 수행.
    - Grayscale: height x width
    - Color: height x width x 3
    Kernel은 홀수 사이즈.
    '''
    if img.ndim == 2:  # Grayscale
        return _cross_correlation_gray(img, kernel)
    elif img.ndim == 3:  # Color 이미지 => 채널별로 따로 적용
        channels = []
        for c in range(img.shape[2]):
            channels.append(_cross_correlation_gray(img[:, :, c], kernel))
        return np.stack(channels, axis=2)
    else:
        raise ValueError("지원되지 않는 이미지 형태")


def _cross_correlation_gray(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Padding (제로 패딩)
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    out = np.zeros_like(img, dtype=np.float32)

    # Correlation = element-wise 곱 후 합
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kernel)

    return out


def convolve_2d(img, kernel):
    '''
    Convolution = cross-correlation with flipped kernel
    '''
    flipped_kernel = np.flipud(np.fliplr(kernel))
    return cross_correlation_2d(img, flipped_kernel)


def low_pass(img, sigma, size):
    '''
    Gaussian low-pass filter.
    '''
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return cross_correlation_2d(img, kernel)


def high_pass(img, sigma, size):
    '''
    High-pass filter = 원본 - Low-pass(blurred).
    '''
    low_img = low_pass(img, sigma, size)
    return img - low_img


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''주어진 파라미터로 두 이미지를 합쳐 Hybrid Image 생성'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    # 두 이미지 합치기
    img1 *= (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
