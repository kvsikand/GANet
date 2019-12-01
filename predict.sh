
python predict.py --crop_height=192 \
                  --crop_width=624 \
                  --max_disp=192 \
                  --data_path='../data_scene_flow/testing/' \
                  --test_list='lists/single_test.list' \
                  --save_path='./result/' \
                  --kitti2015=1 \
                  --resume='./trained_models/kitti2015_final.pth'
exit

python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='/media/feihu/Storage/stereo/kitti/testing/' \
                  --test_list='lists/kitti2012_test.list' \
                  --save_path='./result/' \
                  --kitti=1 \
                  --resume='./checkpoint/kitti2012_final.pth'



