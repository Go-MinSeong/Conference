def choice(Shoes_Categories, name, Parts_Categories, mask_, Style_Categories1, Style_Categories2, theme):
  shoes = name
  mask_choice = mask_
  style1 = theme
  original = cv2.imread(f'shoes/{shoes}.jpg')
  filename=[]

  for i,mask_name in enumerate(mask_choice):
    if i == 0:
      filename1 = f'mask/segmentation_{shoes}_{mask_name}.png'
      merge_mask = cv2.imread(filename1,cv2.IMREAD_GRAYSCALE)
    else:
      filename1 = f'mask/segmentation_{shoes}_{mask_name}.png'
      mask = cv2.imread(filename1,cv2.IMREAD_GRAYSCALE)
      merge_mask = merge_mask + mask

  #바꾸고자 하는 style 이미지

  filename2 = f'style/{style1}.jpg'
  filename3= f'shoes/{shoes}.jpg'
  command = f"python CAP-VSTNet/image_transfer.py --mode photorealistic --ckpoint CAP-VSTNet/checkpoints/photo_image.pt --content {filename3}  --style {filename2} "
  subprocess.run(command, shell=True)
  print('suce\cess')

   # Composition
  #merge_mask = cv2.imread(filename1,cv2.IMREAD_GRAYSCALE)
  output_mask_style = cv2.imread(f'output/{shoes}_{style1}.png')
  transfer = cv2.copyTo(output_mask_style,merge_mask,original) #3개 이미지 곱 style, mask, original
  img_name = f'results/output_mask_{shoes}_{style1}.png'

  NAME = f'results/output_mask_{shoes}_{style1}'

  cv2.imwrite(f'{NAME}.png',transfer)


  # Composition -1
  # mask = cv2.imread(filename1,cv2.IMREAD_GRAYSCALE)
  # output_mask_style = cv2.imread(f'output/segmentation_{shoes}_{mask_choice}_{style1}.png')
  # transfer = cv2.copyTo(output_mask_style,mask,original)
  # img_name = f'results/output_mask_{shoes}_{style1}_{mask_choice}.png'

  # NAME = f'results/output_mask_{shoes}_{style1}_{mask_choice}'

  # cv2.imwrite(f'{NAME}.png',transfer)

  # 3D preprocessing

  command = f"python dreamgaussian/process.py {NAME}.png"

  subprocess.run(command, shell=True)

  # 3D multi view

  client = Client("https://one-2-3-45-one-2-3-45.hf.space/")

  input_img_path = f'{NAME}_rgba.png'

  elevation_angle_deg = client.predict(
    input_img_path,
    True,		# image preprocessing
    api_name="/estimate_elevation"
  )


  Elevation = elevation_angle_deg


  # 3D reconstruction

  command = f"python dreamgaussian/main.py --config dreamgaussian/configs/image.yaml input={input_img_path} save_path={NAME} elevation={Elevation} force_cuda_rast=True"
  subprocess.run(command, shell=True)

  command = f"python dreamgaussian/main2.py --config dreamgaussian/configs/image.yaml input={input_img_path} save_path={NAME} elevation={Elevation} force_cuda_rast=True"
  subprocess.run(command, shell=True)

  #3D obj

  command = f"python -m kiui.render logs/{NAME}.obj --save_video {NAME}.mp4 --wogui --force_cuda_rast"
  subprocess.run(command, shell=True)

  video_name = f'{NAME}.mp4'

  return img_name , video_name