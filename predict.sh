
python predict.py --crop_height=192 \
                  --crop_width=624 \
                  --max_disp=192 \
                  --data_path='../data_scene_flow/testing/' \
                  --test_list='lists/single_test.list' \
                  --save_path='./result/' \
                  --kitti2015=1 \
		  --resume='./trained_models/kitti2015_final.pth' \
		  --noise='homography'
exit
