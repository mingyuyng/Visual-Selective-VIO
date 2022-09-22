#!/bin/sh

wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip    # download the KITTI ODOMETRY dataset
unzip data_odometry_color.zip
mkdir sequences 
mv dataset/sequences/* sequences/
rm -r dataset

for i in {11..21}
do
	rm -r 'sequences/'$i
done

wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip    # download the KITTI ODOMETRY groundtruth poses
unzip data_odometry_poses.zip
mkdir poses
mv dataset/poses/* poses/
rm -r dataset

rm data_odometry_color.zip
rm data_odometry_poses.zip
