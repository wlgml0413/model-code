train_noisy_data = np.load('train_imgs.npy')
train_mask = np.load('train_mask.npy')
val_noisy_data = np.load('validation_imgs.npy')
val_mask = np.load('validation_mask.npy')
test_noisy_data = np.load('test_imgs.npy')
test_mask = np.load('test_mask.npy')

print(train_noisy_data.shape)
print(train_mask.shape)
print(val_noisy_data.shape)
print(val_mask.shape)
print(test_noisy_data.shape)
print(test_mask.shape)

train_noisy_img = train_noisy_data[0, :, :]
train_img = train_mask[0, :, :]
val_noisy_img = val_noisy_data[0, :, :]
val_img = val_mask[0, :, :]
test_noisy_img = test_noisy_data[0, :, :]
test_img = test_mask[0, :, :]

cv2_imshow(train_noisy_img)
cv2_imshow(train_img)
cv2_imshow(val_noisy_img)
cv2_imshow(val_img)
cv2_imshow(test_noisy_img)
cv2_imshow(test_img)

print(train_noisy_data.shape)
print(train_mask.shape)
print(val_noisy_data.shape)
print(val_mask.shape)
print(test_noisy_data.shape)
print(test_mask.shape)

train_noisy_data = train_noisy_data.astype("float32") / 255.0
train_mask = train_mask.astype("float32") / 255.0
val_noisy_data = val_noisy_data.astype("float32") / 255.0
val_mask = val_mask.astype("float32") / 255.0
test_noisy_data = test_noisy_data.astype("float32") / 255.0
test_mask = test_mask.astype("float32") / 255.0

print(train_noisy_data.shape)
print(train_mask.shape)
print(val_noisy_data.shape)
print(val_mask.shape)
print(test_noisy_data.shape)
print(test_mask.shape)
