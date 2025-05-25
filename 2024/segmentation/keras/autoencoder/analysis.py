import time

# 추론 시간 측정
start_time = time.time()
imgs_test_pred = model_1.predict(test_noisy_data)
end_time = time.time()

# 이미지 디스플레이
display(test_noisy_data, imgs_test_pred)
np.save('/content/drive/MyDrive/data/npy_best_friend/imgs_test_pred_autoencoder.npy', imgs_test_pred)
print(test_noisy_data.shape)
print(imgs_test_pred.shape)

# 시간 계산 (ms 단위)
total_time = (end_time - start_time) * 1000  # 전체 추론 시간(ms)
average_time_per_inference = total_time / len(test_noisy_data)  # 영상 1장당 평균 시간(ms)

# 결과 출력
print(f"Total Inference Time: {total_time:.3f} ms")
print(f"Time per Inference Step (ms): {average_time_per_inference:.3f} ms")


size = test_noisy_data.shape[0] // 5
w = test_noisy_data.shape[2]
h = test_noisy_data.shape[1]
t_img = np.zeros((3 * h, 5 * w), np.uint8)
for i in range(5):
    t_img[0:h, i * w:i * w + w] = 255 * test_noisy_data[i,:,:]
    t_img[h:h + h, i * w:i * w + w] = 255 * test_mask[i,:,:]
    t_img[h + h:h + h + h, i * w:i * w + w] = 255 * imgs_test_pred[i,:,:]
# cv2.putText(t_img, 'noise o', (0, 20), 2, 1, (192, 192, 192))
# cv2.putText(t_img, 'noise x', (0, 256 + 20), 2, 1, (192, 192, 192))
# cv2.putText(t_img, 'result', (0, 512 + 20), 2, 1, (192, 192, 192))

cv2_imshow(t_img)


from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_squared_error

psnr_value = peak_signal_noise_ratio(test_mask, imgs_test_pred)
ssim_loss = structural_similarity(test_mask, imgs_test_pred, channel_axis=-1, data_range=test_mask.max()-test_mask.min())
mse_value = mean_squared_error(test_mask.flatten(), imgs_test_pred.flatten())

print("**이미지 비교**")
print("PSNR :", psnr_value)
print("SSIM :", ssim_loss)
print("MSE :", mse_value)
