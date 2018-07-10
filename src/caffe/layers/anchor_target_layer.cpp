#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/anchor_target_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

using std::sqrt;
using std::floor;
using std::ceil;
using std::min;
using std::max;
using std::log;

namespace caffe {

template <typename Dtype>
void AnchorTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const unsigned int rng_seed = caffe_rng_rand();
	rng_.reset(new Caffe::RNG(rng_seed));

	const AnchorTargetParameter& anchor_target_param = this->layer_param_.anchor_target_param();
	const AnchorParameter& anchor_param = anchor_target_param.anchor_size();
	anchors_ = caffe::GenerateAnchors(anchor_param, anchors_num_);
	feat_size_[0] = bottom[0]->shape(4);
	feat_size_[1] = bottom[0]->shape(3);
	feat_size_[2] = bottom[0]->shape(2);
	vector<int> top_shape(5);
	top_shape[0] = 1;
	top_shape[1] = 1;
	top_shape[2] = anchors_num_ * feat_size_[2];
	top_shape[3] = feat_size_[1];
	top_shape[4] = feat_size_[0];
	top[0]->Reshape(top_shape);
	top_shape[1] = anchors_num_ * 6;
	top_shape[2] = feat_size_[2];
	top[1]->Reshape(top_shape);
	top[2]->Reshape(top_shape);
	top[3]->Reshape(top_shape);

	bbox_target_num_ = 0;
	for (int i = 0; i < 6; ++i)
	{
		bbox_target_mean_[i] = 0;
		bbox_target_mean_2_[i] = 0;
	}
}

template <typename Dtype>
void AnchorTargetLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
  
}

template <typename Dtype>
void AnchorTargetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	const int batch_size = bottom[0]->shape(0);
	feat_size_[0] = bottom[0]->shape(4);
	feat_size_[1] = bottom[0]->shape(3);
	feat_size_[2] = bottom[0]->shape(2);

	const int all_gt_box_num = bottom[1]->shape(1);

	vector<int> top_shape(5);
	top_shape[0] = batch_size;
	top_shape[1] = 1;
	top_shape[2] = anchors_num_ * feat_size_[2];
	top_shape[3] = feat_size_[1];
	top_shape[4] = feat_size_[0];
	top[0]->Reshape(top_shape);

	top_shape[0] = batch_size;
	top_shape[1] = anchors_num_ * 6;
	top_shape[2] = feat_size_[2];
	top_shape[3] = feat_size_[1];
	top_shape[4] = feat_size_[0];
	top[1]->Reshape(top_shape);
	top[2]->Reshape(top_shape);
	top[3]->Reshape(top_shape);

	const AnchorTargetParameter& anchor_target_param = this->layer_param_.anchor_target_param();
	const bool use_multi_classification = anchor_target_param.use_multi_classification();

	// generate all ref_boxes
	const int feat_stride_xy = anchor_target_param.feat_stride_xy();
	const int feat_stride_z = anchor_target_param.feat_stride_z();
	const int all_ref_box_num = feat_size_[2] * feat_size_[1] * feat_size_[0] * anchors_num_;
	Dtype* all_ref_box = new Dtype[all_ref_box_num * 6];
#pragma omp parallel for
	for (int shift_z = 0; shift_z < feat_size_[2]; ++shift_z) {
		for (int shift_y = 0; shift_y < feat_size_[1]; ++shift_y) {
			for (int shift_x = 0; shift_x < feat_size_[0]; ++shift_x) {
				for (int m = 0; m < anchors_num_; ++m) {
					all_ref_box[(((shift_z * feat_size_[1] + shift_y) * feat_size_[0] + shift_x) * anchors_num_ + m) * 6 + 0] =
						shift_x * feat_stride_xy + anchors_[m * 6 + 0];
					all_ref_box[(((shift_z * feat_size_[1] + shift_y) * feat_size_[0] + shift_x) * anchors_num_ + m) * 6 + 1] =
						shift_y * feat_stride_xy + anchors_[m * 6 + 1];
					all_ref_box[(((shift_z * feat_size_[1] + shift_y) * feat_size_[0] + shift_x) * anchors_num_ + m) * 6 + 2] =
						shift_z * feat_stride_z + anchors_[m * 6 + 2];

					all_ref_box[(((shift_z * feat_size_[1] + shift_y) * feat_size_[0] + shift_x) * anchors_num_ + m) * 6 + 3] =
						shift_x * feat_stride_xy + anchors_[m * 6 + 3];
					all_ref_box[(((shift_z * feat_size_[1] + shift_y) * feat_size_[0] + shift_x) * anchors_num_ + m) * 6 + 4] =
						shift_y * feat_stride_xy + anchors_[m * 6 + 4];
					all_ref_box[(((shift_z * feat_size_[1] + shift_y) * feat_size_[0] + shift_x) * anchors_num_ + m) * 6 + 5] =
						shift_z * feat_stride_z + anchors_[m * 6 + 5];
				}
			}
		}
	}

	// only keep ref_boxes inside the image
	const Dtype* im_info = bottom[2]->cpu_data() + 19 * 0; // first image info
	int inds_inside_num = 0;
	int* tmp_buff = new int[all_ref_box_num];
	for (int i = 0; i < all_ref_box_num; ++i) {
		if (all_ref_box[i * 6 + 0] >= 0 &&
			all_ref_box[i * 6 + 1] >= 0 &&
			all_ref_box[i * 6 + 2] >= 0 &&
			all_ref_box[i * 6 + 3] < im_info[2] &&
			all_ref_box[i * 6 + 4] < im_info[1] &&
			all_ref_box[i * 6 + 5] < im_info[0]) {
			tmp_buff[inds_inside_num++] = i;
		}
	}
	int* inds_inside = new int[inds_inside_num];
	memcpy(inds_inside, tmp_buff, sizeof(int) * inds_inside_num);
	delete[]tmp_buff;

	Dtype* inside_ref_box = new Dtype[inds_inside_num * 6];
	for (int i = 0; i < inds_inside_num; ++i) {
		for (int j = 0; j < 6; ++j) {
			inside_ref_box[i * 6 + j] = all_ref_box[inds_inside[i] * 6 + j];
		}
	}
	delete[]all_ref_box;

	int* label = new int[inds_inside_num];	// label: positive number is foreground, 0 is background, -1 is don't care
	Dtype* bbox_targets = new Dtype[inds_inside_num * 6];

	for (int batch_id = 0; batch_id < batch_size; ++batch_id) {

		// calculate enabled gt_box number
		const Dtype* gt_box_src = bottom[1]->cpu_data() + batch_id * all_gt_box_num * 8;
		int gt_box_num = 0;
		for (int i = 0; i < all_gt_box_num; ++i)
		{
			if (gt_box_src[i * 8 + 7] > 0)
			{
				gt_box_num++;
			}
		}

		// assign labels to each ref_box inside
		if (gt_box_num > 0)
		{
			// count gt_boxes enabled
			Dtype* gt_box = new Dtype[gt_box_num * 7];
			gt_box_num = 0;
			for (int i = 0; i < all_gt_box_num; ++i)
			{
				if (gt_box_src[i * 8 + 7] > 0)
				{
					for (int j = 0; j < 7; ++j)
					{
						gt_box[gt_box_num * 7 + j] = gt_box_src[i * 8 + j];
					}
					if (!use_multi_classification)
					{
						gt_box[gt_box_num * 7 + 6] = 1; // assign all positive labels to foreground (1)
					}
					gt_box_num++;
				}
			}

			// calculate IoU overlaps between each gt_box and ref_box
			double* overlaps = new double[gt_box_num * inds_inside_num];
			double overlap_max = -1.0;
			double overlap_min = 1.0;
	#pragma omp parallel for
			for (int i = 0; i < inds_inside_num; ++i) {
				for (int j = 0; j < gt_box_num; ++j) {
					overlaps[j * inds_inside_num + i] = 0;
					double intersection_size[3];
					intersection_size[0] =
						min(gt_box[j * 7 + 3], inside_ref_box[i * 6 + 3]) -
						max(gt_box[j * 7 + 0], inside_ref_box[i * 6 + 0]) + 1;
					if (intersection_size[0] > 0) {
						intersection_size[1] =
							min(gt_box[j * 7 + 4], inside_ref_box[i * 6 + 4]) -
							max(gt_box[j * 7 + 1], inside_ref_box[i * 6 + 1]) + 1;
						if (intersection_size[1] > 0) {
							intersection_size[2] =
								min(gt_box[j * 7 + 5], inside_ref_box[i * 6 + 5]) -
								max(gt_box[j * 7 + 2], inside_ref_box[i * 6 + 2]) + 1;
							if (intersection_size[2] > 0) {
								double anchor_volume =
									(inside_ref_box[i * 6 + 3] - inside_ref_box[i * 6 + 0] + 1) *
									(inside_ref_box[i * 6 + 4] - inside_ref_box[i * 6 + 1] + 1) *
									(inside_ref_box[i * 6 + 5] - inside_ref_box[i * 6 + 2] + 1);
								double gt_box_volume =
									(gt_box[j * 7 + 3] - gt_box[j * 7 + 0] + 1) *
									(gt_box[j * 7 + 4] - gt_box[j * 7 + 1] + 1) *
									(gt_box[j * 7 + 5] - gt_box[j * 7 + 2] + 1);
								double intersection_volume =
									intersection_size[0] * intersection_size[1] * intersection_size[2];
								overlaps[j * inds_inside_num + i] =
									intersection_volume / (anchor_volume + gt_box_volume - intersection_volume);
								if (overlap_max < overlaps[j * inds_inside_num + i]) {
									overlap_max = overlaps[j * inds_inside_num + i];
								}
								if (overlap_min > overlaps[j * inds_inside_num + i]) {
									overlap_min = overlaps[j * inds_inside_num + i];
								}
							}
						}
					}
				}
			}
			//LOG(INFO) << "overlap: min = " << overlap_min << " max = " << overlap_max;
			int* argmax_overlaps = new int[inds_inside_num];
			double* max_overlaps = new double[inds_inside_num];
			for (int i = 0; i < inds_inside_num; ++i) {
				argmax_overlaps[i] = 0;
				max_overlaps[i] = overlaps[0 * inds_inside_num + i];
				for (int j = 1; j < gt_box_num; ++j) {
					if (overlaps[j * inds_inside_num + i] > max_overlaps[i]) {
						argmax_overlaps[i] = j;
						max_overlaps[i] = overlaps[j * inds_inside_num + i];
					}
				}
			}
			int* gt_argmax_overlaps = new int[gt_box_num];
			double* gt_max_overlaps = new double[gt_box_num];
			char* picked = new char[inds_inside_num];
			memset(picked, 0, sizeof(char) * inds_inside_num);
			for (int i = 0; i < gt_box_num; ++i) {
				gt_argmax_overlaps[i] = 0;
				gt_max_overlaps[i] = overlaps[i * inds_inside_num + 0];
				for (int j = 1; j < inds_inside_num; ++j) {
					if (overlaps[i * inds_inside_num + j] > gt_max_overlaps[i] && !picked[j]) {
						gt_argmax_overlaps[i] = j;
						gt_max_overlaps[i] = overlaps[i * inds_inside_num + j];
					}
				}
				picked[gt_argmax_overlaps[i]] = 1;
			}
			delete[]picked;
			delete[]overlaps;

			for (int i = 0; i < inds_inside_num; ++i) {
				label[i] = -1;
			}

			if (!anchor_target_param.rpn_clobber_positives()) {
				// assign bg labels first so that positive labels can clobber them
				for (int i = 0; i < inds_inside_num; ++i) {
					if (max_overlaps[i] < anchor_target_param.rpn_negative_overlap()) {
						label[i] = 0;
					}
				}
			}
			// fg label: for each gt, anchor with highest overlap
			for (int i = 0; i < gt_box_num; ++i) {
				label[gt_argmax_overlaps[i]] = gt_box[i * 7 + 6];
			}
			// fg label: above threshold IoU
			for (int i = 0; i < inds_inside_num; ++i) {
				if (max_overlaps[i] > anchor_target_param.rpn_positive_overlap()) {
					label[i] = gt_box[argmax_overlaps[i] * 7 + 6];
				}
			}
			if (anchor_target_param.rpn_clobber_positives()) {
				// assign bg labels last so that negative labels can clobber positives
				for (int i = 0; i < inds_inside_num; ++i) {
					if (max_overlaps[i] < anchor_target_param.rpn_negative_overlap()) {
						label[i] = 0;
					}
				}
			}

			for (int i = 0; i < inds_inside_num; ++i) {
				if (label[i] <= 0)
				{
					bbox_targets[i * 6 + 0] = 0;
					bbox_targets[i * 6 + 1] = 0;
					bbox_targets[i * 6 + 2] = 0;
					bbox_targets[i * 6 + 3] = 0;
					bbox_targets[i * 6 + 4] = 0;
					bbox_targets[i * 6 + 5] = 0;
				}
				else
				{
					Dtype ref_box_size[3];
					Dtype ref_box_center[3];
					ref_box_size[0] = inside_ref_box[i * 6 + 3] - inside_ref_box[i * 6 + 0] + 1.0;
					ref_box_size[1] = inside_ref_box[i * 6 + 4] - inside_ref_box[i * 6 + 1] + 1.0;
					ref_box_size[2] = inside_ref_box[i * 6 + 5] - inside_ref_box[i * 6 + 2] + 1.0;
					ref_box_center[0] = inside_ref_box[i * 6 + 0] + 0.5 * ref_box_size[0];
					ref_box_center[1] = inside_ref_box[i * 6 + 1] + 0.5 * ref_box_size[1];
					ref_box_center[2] = inside_ref_box[i * 6 + 2] + 0.5 * ref_box_size[2];
					Dtype gt_box_size[3];
					Dtype gt_box_center[3];
					gt_box_size[0] = gt_box[argmax_overlaps[i] * 7 + 3] - gt_box[argmax_overlaps[i] * 7 + 0] + 1.0;
					gt_box_size[1] = gt_box[argmax_overlaps[i] * 7 + 4] - gt_box[argmax_overlaps[i] * 7 + 1] + 1.0;
					gt_box_size[2] = gt_box[argmax_overlaps[i] * 7 + 5] - gt_box[argmax_overlaps[i] * 7 + 2] + 1.0;
					gt_box_center[0] = gt_box[argmax_overlaps[i] * 7 + 0] + 0.5 * gt_box_size[0];
					gt_box_center[1] = gt_box[argmax_overlaps[i] * 7 + 1] + 0.5 * gt_box_size[1];
					gt_box_center[2] = gt_box[argmax_overlaps[i] * 7 + 2] + 0.5 * gt_box_size[2];

					bbox_targets[i * 6 + 0] = (gt_box_center[0] - ref_box_center[0]) / ref_box_size[0];
					bbox_targets[i * 6 + 1] = (gt_box_center[1] - ref_box_center[1]) / ref_box_size[1];
					bbox_targets[i * 6 + 2] = (gt_box_center[2] - ref_box_center[2]) / ref_box_size[2];
					bbox_targets[i * 6 + 3] = log(gt_box_size[0] / ref_box_size[0]);
					bbox_targets[i * 6 + 4] = log(gt_box_size[1] / ref_box_size[1]);
					bbox_targets[i * 6 + 5] = log(gt_box_size[2] / ref_box_size[2]);

					bbox_target_num_++;
					for (int j = 0; j < 6; ++j)
					{
						bbox_target_mean_[j] += bbox_targets[i * 6 + j];
						bbox_target_mean_2_[j] += bbox_targets[i * 6 + j] * bbox_targets[i * 6 + j];
					}

					if (anchor_target_param.bbox_normalize())
					{
						for (int j = 0; j < 6; ++j)
						{
							bbox_targets[i * 6 + j] = (bbox_targets[i * 6 + j] - anchor_target_param.bbox_normalize_mean(j)) / anchor_target_param.bbox_normalize_std(j);
						}
					}
				}
			}
			delete[]argmax_overlaps;
			delete[]max_overlaps;
			delete[]gt_argmax_overlaps;
			delete[]gt_max_overlaps;
			delete[]gt_box;
		}
		else
		{
			// if there is no gt_box enabled, we assign all ref_boxes to background
			for (int i = 0; i < inds_inside_num; ++i) {
				label[i] = 0;
				bbox_targets[i * 6 + 0] = 0;
				bbox_targets[i * 6 + 1] = 0;
				bbox_targets[i * 6 + 2] = 0;
				bbox_targets[i * 6 + 3] = 0;
				bbox_targets[i * 6 + 4] = 0;
				bbox_targets[i * 6 + 5] = 0;
			}
		}

		int fg_num_exist = 0;
		for (int i = 0; i < inds_inside_num; ++i) {
			if (label[i] > 0) {
				fg_num_exist++;
			}
		}
		//LOG(INFO) << "foreground found: " << fg_num_exist;

		if (fg_num_exist > 0)
		{
			for (int i = 0; i < 6; ++i)
			{
				double mean = bbox_target_mean_[i] / bbox_target_num_;
				double mean_2 = bbox_target_mean_2_[i] / bbox_target_num_;
				LOG(INFO) << i << ": mean = " << mean << " std = " << sqrt(mean_2 - mean * mean) << " num = " << bbox_target_num_;
			}
		}

		int bg_num_exist = 0;
		for (int i = 0; i < inds_inside_num; ++i) {
			if (label[i] == 0) {
				bg_num_exist++;
			}
		}
		//LOG(INFO) << "background found: " << bg_num_exist;

		Dtype* bbox_inside_weights = new Dtype[inds_inside_num * 6];
		for (int i = 0; i < inds_inside_num; ++i) {
			if (label[i] > 0) {
				for (int j = 0; j < 6; ++j) {
					bbox_inside_weights[i * 6 + j] = 1;
				}
			}
			else {
				for (int j = 0; j < 6; ++j) {
					bbox_inside_weights[i * 6 + j] = 0;
				}
			}
		}

		Dtype* bbox_outside_weights = new Dtype[inds_inside_num * 6];
		int num_examples = fg_num_exist + bg_num_exist;
		for (int i = 0; i < inds_inside_num; ++i) {
			if (label[i] > 0) {
				for (int j = 0; j < 6; ++j) {
					bbox_outside_weights[i * 6 + j] = 1.0 / fg_num_exist;
				}
			}
			else if (label[i] == 0)
			{
				for (int j = 0; j < 6; ++j) {
					bbox_outside_weights[i * 6 + j] = 1.0 / bg_num_exist;
				}
			}
			else {
				for (int j = 0; j < 6; ++j) {
					bbox_outside_weights[i * 6 + j] = 0;
				}
			}
		}
  
		// map up to original set of ref_box
		int* all_label = new int[all_ref_box_num];
		for (int i = 0; i < all_ref_box_num; ++i) {
			all_label[i] = -1;
		}
		for (int i = 0; i < inds_inside_num; ++i) {
			all_label[inds_inside[i]] = label[i];
		}

		Dtype* all_bbox_targets = new Dtype[all_ref_box_num * 6];
		for (int i = 0; i < all_ref_box_num * 6; ++i) {
			all_bbox_targets[i] = 0;
		}
		for (int i = 0; i < inds_inside_num; ++i) {
			for (int j = 0; j < 6; ++j) {
				all_bbox_targets[inds_inside[i] * 6 + j] = bbox_targets[i * 6 + j];
			}
		}

		Dtype* all_bbox_inside_weights = new Dtype[all_ref_box_num * 6];
		for (int i = 0; i < all_ref_box_num * 6; ++i) {
			all_bbox_inside_weights[i] = 0;
		}
		for (int i = 0; i < inds_inside_num; ++i) {
			for (int j = 0; j < 6; ++j) {
				all_bbox_inside_weights[inds_inside[i] * 6 + j] = bbox_inside_weights[i * 6 + j];
			}
		}
		delete[]bbox_inside_weights;

		Dtype* all_bbox_outside_weights = new Dtype[all_ref_box_num * 6];
		for (int i = 0; i < all_ref_box_num * 6; ++i) {
			all_bbox_outside_weights[i] = 0;
		}
		for (int i = 0; i < inds_inside_num; ++i) {
			for (int j = 0; j < 6; ++j) {
				all_bbox_outside_weights[inds_inside[i] * 6 + j] = bbox_outside_weights[i * 6 + j];
			}
		}
		delete[]bbox_outside_weights;

		Dtype* top_0_data = top[0]->mutable_cpu_data() + batch_id * all_ref_box_num;
		for (int n = 0; n < anchors_num_; ++n) {
			for (int i = 0; i < feat_size_[2]; ++i) {
				for (int j = 0; j < feat_size_[1]; ++j) {
					for (int k = 0; k < feat_size_[0]; ++k) {
						top_0_data[((n * feat_size_[2] + i) * feat_size_[1] + j) * feat_size_[0] + k] =
							all_label[((i * feat_size_[1] + j) * feat_size_[0] + k) * anchors_num_ + n];
					}
				}
			}
		}
		delete[]all_label;

		Dtype* top_1_data = top[1]->mutable_cpu_data() + batch_id * all_ref_box_num * 6;
		Dtype* top_2_data = top[2]->mutable_cpu_data() + batch_id * all_ref_box_num * 6;
		Dtype* top_3_data = top[3]->mutable_cpu_data() + batch_id * all_ref_box_num * 6;
		for (int n = 0; n < anchors_num_; ++n) {
			for (int m = 0; m < 6; ++m) {
				for (int i = 0; i < feat_size_[2]; ++i) {
					for (int j = 0; j < feat_size_[1]; ++j) {
						for (int k = 0; k < feat_size_[0]; ++k) {
							top_1_data[(((n * 6 + m) * feat_size_[2] + i) * feat_size_[1] + j) * feat_size_[0] + k] =
								all_bbox_targets[(((i * feat_size_[1] + j) * feat_size_[0] + k) * anchors_num_ + n) * 6 + m];
							top_2_data[(((n * 6 + m) * feat_size_[2] + i) * feat_size_[1] + j) * feat_size_[0] + k] =
								all_bbox_inside_weights[(((i * feat_size_[1] + j) * feat_size_[0] + k) * anchors_num_ + n) * 6 + m];
							top_3_data[(((n * 6 + m) * feat_size_[2] + i) * feat_size_[1] + j) * feat_size_[0] + k] =
								all_bbox_outside_weights[(((i * feat_size_[1] + j) * feat_size_[0] + k) * anchors_num_ + n) * 6 + m];
						}
					}
				}
			}
		}
		delete[]all_bbox_targets;
		delete[]all_bbox_inside_weights;
		delete[]all_bbox_outside_weights;
	}

	delete[]inside_ref_box;
	delete[]label;
	delete[]bbox_targets;
	delete[]inds_inside;
}

template <typename Dtype>
void AnchorTargetLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(AnchorTargetLayer);
REGISTER_LAYER_CLASS(AnchorTarget);

}  // namespace caffe