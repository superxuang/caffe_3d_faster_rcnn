#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/proposal_target_layer.hpp"
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
void ProposalTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const unsigned int rng_seed = caffe_rng_rand();
	rng_.reset(new Caffe::RNG(rng_seed));
	const ProposalTargetParameter& proposal_target_param = this->layer_param_.proposal_target_param();
	vector<int> top_shape(2);
	top_shape[0] = 1;
	top_shape[1] = 7;
	top[0]->Reshape(top_shape);
	top_shape[0] = 1;
	top_shape[1] = 1;
	top[1]->Reshape(top_shape);
	top_shape[0] = 1;
	top_shape[1] = proposal_target_param.class_num() * 6;
	top[2]->Reshape(top_shape);
	top[3]->Reshape(top_shape);
	top[4]->Reshape(top_shape);

	bbox_target_num_ = 0;
	for (int i = 0; i < 6; ++i)
	{
		bbox_target_mean_[i] = 0;
		bbox_target_mean_2_[i] = 0;
	}
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
  
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	const ProposalTargetParameter& proposal_target_param = this->layer_param_.proposal_target_param();
	int all_rois_num = bottom[0]->shape(0);
	const int batch_size = bottom[1]->shape(0);
	const int all_gt_box_num = bottom[1]->shape(1);
	const int class_num = proposal_target_param.class_num();
	const Dtype* rois_src = bottom[0]->cpu_data();

	int* batch_rois_num = new int[batch_size];
	Dtype** sample_rois = new Dtype*[batch_size];
	Dtype** sample_labels = new Dtype*[batch_size];
	Dtype** bbox_targets = new Dtype*[batch_size];
	Dtype** bbox_inside_weights = new Dtype*[batch_size];
	Dtype** bbox_outside_weights = new Dtype*[batch_size];

	for (int batch_id = 0; batch_id < batch_size; ++batch_id)
	{
		// calculate enabled gt_box number
		const Dtype* gt_box_src = bottom[1]->cpu_data() + batch_id * all_gt_box_num * 8;
		int gt_box_num = 0;
		Dtype* gt_box = NULL;
		for (int i = 0; i < all_gt_box_num; ++i)
		{
			if (gt_box_src[i * 8 + 7] > 0)
			{
				gt_box_num++;
			}
		}

		if (gt_box_num > 0)
		{
			// count gt_boxes enabled
			gt_box = new Dtype[gt_box_num * 7];
			gt_box_num = 0;
			for (int i = 0; i < all_gt_box_num; ++i)
			{
				if (gt_box_src[i * 8 + 7] > 0)
				{
					for (int j = 0; j < 7; ++j)
					{
						gt_box[gt_box_num * 7 + j] = gt_box_src[i * 8 + j];
					}
					gt_box_num++;
				}
			}
		}

		int rois_num = 0;
		for (int i = 0; i < all_rois_num; ++i)
		{
			if (rois_src[i * 7 + 0] == batch_id)
			{
				rois_num++;
			}
		}
		Dtype* rois = new Dtype[rois_num * 7];
		rois_num = 0;
		for (int i = 0; i < all_rois_num; ++i)
		{
			if (rois_src[i * 7 + 0] == batch_id)
			{
				for (int j = 0; j < 7; ++j) 
				{
					rois[rois_num * 7 + j] = rois_src[i * 7 + j];
				}
				rois_num++;
			}
		}
		rois_num = rois_num + gt_box_num;
		Dtype* all_rois = new Dtype[rois_num * 7];
		for (int i = 0; i < gt_box_num; ++i) {
			all_rois[i * 7 + 0] = batch_id;
			for (int j = 0; j < 6; ++j) {
				all_rois[i * 7 + 1 + j] = gt_box[i * 7 + j];
			}
		}
		for (int i = gt_box_num; i < rois_num; ++i) {
			for (int j = 0; j < 7; ++j) {
				all_rois[i * 7 + j] = rois[(i - gt_box_num) * 7 + j];
			}
		}

		batch_rois_num[batch_id] = rois_num;
		int* labels = new int[rois_num];
		memset(labels, 0, sizeof(int) * rois_num);
		int fg_num = 0;
		int bg_num = 0;
		int* fg_inds = new int[rois_num];
		int* bg_inds = new int[rois_num];
		memset(fg_inds, 0, sizeof(int) * rois_num);
		memset(bg_inds, 0, sizeof(int) * rois_num);

		int* argmax_overlaps = new int[rois_num];
		double* max_overlaps = new double[rois_num];

		if (gt_box_num > 0)
		{
			double* overlaps = new double[gt_box_num * rois_num];
			double overlap_max = -1;
			double overlap_min = 1;
#pragma omp parallel for
			for (int i = 0; i < rois_num; ++i) {
				for (int j = 0; j < gt_box_num; ++j) {
					overlaps[j * rois_num + i] = 0;
					double intersection_size[3];
					intersection_size[0] =
						min(gt_box[j * 7 + 3], all_rois[i * 7 + 1 + 3]) -
						max(gt_box[j * 7 + 0], all_rois[i * 7 + 1 + 0]) + 1;
					if (intersection_size[0] > 0) {
						intersection_size[1] =
							min(gt_box[j * 7 + 4], all_rois[i * 7 + 1 + 4]) -
							max(gt_box[j * 7 + 1], all_rois[i * 7 + 1 + 1]) + 1;
						if (intersection_size[1] > 0) {
							intersection_size[2] =
								min(gt_box[j * 7 + 5], all_rois[i * 7 + 1 + 5]) -
								max(gt_box[j * 7 + 2], all_rois[i * 7 + 1 + 2]) + 1;
							if (intersection_size[2] > 0) {
								double roi_volume =
									(all_rois[i * 7 + 1 + 3] - all_rois[i * 7 + 1 + 0] + 1) *
									(all_rois[i * 7 + 1 + 4] - all_rois[i * 7 + 1 + 1] + 1) *
									(all_rois[i * 7 + 1 + 5] - all_rois[i * 7 + 1 + 2] + 1);
								double gt_box_volume =
									(gt_box[j * 7 + 3] - gt_box[j * 7 + 0] + 1) *
									(gt_box[j * 7 + 4] - gt_box[j * 7 + 1] + 1) *
									(gt_box[j * 7 + 5] - gt_box[j * 7 + 2] + 1);
								double intersection_volume =
									intersection_size[0] * intersection_size[1] * intersection_size[2];
								overlaps[j * rois_num + i] =
									intersection_volume / (roi_volume + gt_box_volume - intersection_volume);
								if (overlap_min > overlaps[j * rois_num + i])
									overlap_min = overlaps[j * rois_num + i];
								if (overlap_max < overlaps[j * rois_num + i])
									overlap_max = overlaps[j * rois_num + i];
							}
						}
					}
				}
			}

			for (int i = 0; i < rois_num; ++i) {
				argmax_overlaps[i] = 0;
				max_overlaps[i] = overlaps[0 * rois_num + i];
				labels[i] = gt_box[0 * 7 + 6];
				for (int j = 1; j < gt_box_num; ++j) {
					if (overlaps[j * rois_num + i] > max_overlaps[i]) {
						argmax_overlaps[i] = j;
						max_overlaps[i] = overlaps[j * rois_num + i];
						labels[i] = gt_box[j * 7 + 6];
					}
				}
				if (max_overlaps[i] >= proposal_target_param.fg_threshold()) {
					fg_inds[fg_num++] = i;
				}
				else {
					bg_inds[bg_num++] = i;
				}
			}
			delete[]overlaps;
		}
		else
		{
			fg_num = 0;
			bg_num = rois_num;
			for (int i = 0; i < rois_num; ++i)
			{
				labels[i] = 0;
				bg_inds[i] = i;
			}
		}
		LOG(INFO) << "fg num = " << fg_num << " bg num = " << bg_num;

		// Select sampled values from various arrays
		sample_rois[batch_id] = new Dtype[rois_num * 7];
		sample_labels[batch_id] = new Dtype[rois_num];
		bbox_targets[batch_id] = new Dtype[rois_num * class_num * 6];
		bbox_inside_weights[batch_id] = new Dtype[rois_num * class_num * 6];
		bbox_outside_weights[batch_id] = new Dtype[rois_num * class_num * 6];
		memset(bbox_targets[batch_id], 0, sizeof(Dtype) * rois_num * class_num * 6);
		memset(bbox_inside_weights[batch_id], 0, sizeof(Dtype) * rois_num * class_num * 6);
		memset(bbox_outside_weights[batch_id], 0, sizeof(Dtype) * rois_num * class_num * 6);

		int* class_instance_num = new int[class_num];
		memset(class_instance_num, 0, sizeof(int) * class_num);
		int* sample_inds = new int[rois_num];
		for (int i = 0; i < fg_num; ++i) {
			sample_inds[i] = fg_inds[i];
			sample_labels[batch_id][i] = labels[fg_inds[i]];
			class_instance_num[labels[fg_inds[i]]]++;
			for (int j = 0; j < 7; ++j) {
				sample_rois[batch_id][i * 7 + j] = all_rois[fg_inds[i] * 7 + j];
			}
		}
		for (int i = 0; i < bg_num; ++i) {
			sample_inds[i + fg_num] = bg_inds[i];
			// Clamp labels for the background RoIs to 0
			sample_labels[batch_id][i + fg_num] = 0;
			for (int j = 0; j < 7; ++j) {
				sample_rois[batch_id][(i + fg_num) * 7 + j] = all_rois[bg_inds[i] * 7 + j];
			}
		}
		delete[]labels;
		delete[]fg_inds;
		delete[]bg_inds;
		delete[]all_rois;
		for (int i = 1; i < class_num; ++i)
		{
			LOG(INFO) << "class " << i << " : " << class_instance_num[i];
		}
		delete[]class_instance_num;

		if (gt_box_num > 0)	
		{
			for (int i = 0; i < rois_num; ++i) 
			{
				int offset = i * class_num * 6 + sample_labels[batch_id][i] * 6;
				if (sample_labels[batch_id][i] > 0)
				{
					Dtype ex_size[3];
					Dtype ex_center[3];
					ex_size[0] = sample_rois[batch_id][i * 7 + 1 + 3] - sample_rois[batch_id][i * 7 + 1 + 0] + 1.0;
					ex_size[1] = sample_rois[batch_id][i * 7 + 1 + 4] - sample_rois[batch_id][i * 7 + 1 + 1] + 1.0;
					ex_size[2] = sample_rois[batch_id][i * 7 + 1 + 5] - sample_rois[batch_id][i * 7 + 1 + 2] + 1.0;
					ex_center[0] = sample_rois[batch_id][i * 7 + 1 + 0] + 0.5 * ex_size[0];
					ex_center[1] = sample_rois[batch_id][i * 7 + 1 + 1] + 0.5 * ex_size[1];
					ex_center[2] = sample_rois[batch_id][i * 7 + 1 + 2] + 0.5 * ex_size[2];
					Dtype gt_size[3];
					Dtype gt_center[3];
					gt_size[0] = gt_box[argmax_overlaps[sample_inds[i]] * 7 + 3] - gt_box[argmax_overlaps[sample_inds[i]] * 7 + 0] + 1.0;
					gt_size[1] = gt_box[argmax_overlaps[sample_inds[i]] * 7 + 4] - gt_box[argmax_overlaps[sample_inds[i]] * 7 + 1] + 1.0;
					gt_size[2] = gt_box[argmax_overlaps[sample_inds[i]] * 7 + 5] - gt_box[argmax_overlaps[sample_inds[i]] * 7 + 2] + 1.0;
					gt_center[0] = gt_box[argmax_overlaps[sample_inds[i]] * 7 + 0] + 0.5 * gt_size[0];
					gt_center[1] = gt_box[argmax_overlaps[sample_inds[i]] * 7 + 1] + 0.5 * gt_size[1];
					gt_center[2] = gt_box[argmax_overlaps[sample_inds[i]] * 7 + 2] + 0.5 * gt_size[2];

					bbox_targets[batch_id][offset + 0] = (gt_center[0] - ex_center[0]) / ex_size[0];
					bbox_targets[batch_id][offset + 1] = (gt_center[1] - ex_center[1]) / ex_size[1];
					bbox_targets[batch_id][offset + 2] = (gt_center[2] - ex_center[2]) / ex_size[2];
					bbox_targets[batch_id][offset + 3] = log(gt_size[0] / ex_size[0]);
					bbox_targets[batch_id][offset + 4] = log(gt_size[1] / ex_size[1]);
					bbox_targets[batch_id][offset + 5] = log(gt_size[2] / ex_size[2]);
					for (int j = 0; j < 6; ++j) {
						bbox_inside_weights[batch_id][offset + j] = 1.0;
						bbox_outside_weights[batch_id][offset + j] = 1.0;
					}

					bbox_target_num_++;
					for (int j = 0; j < 6; ++j)
					{
						bbox_target_mean_[j] += bbox_targets[batch_id][offset + j];
						bbox_target_mean_2_[j] += bbox_targets[batch_id][offset + j] * bbox_targets[batch_id][offset + j];
					}

					if (proposal_target_param.bbox_normalize())
					{
						for (int j = 0; j < 6; ++j)
						{
							bbox_targets[batch_id][offset + j] = (bbox_targets[batch_id][offset + j] - proposal_target_param.bbox_normalize_mean(j)) / proposal_target_param.bbox_normalize_std(j);
						}
					}
				}
			}
		}

		delete[]max_overlaps;
		delete[]argmax_overlaps;
		delete[]sample_inds;
		delete[]gt_box;
	}

	for (int i = 0; i < 6; ++i)
	{
		double mean = bbox_target_mean_[i] / bbox_target_num_;
		double mean_2 = bbox_target_mean_2_[i] / bbox_target_num_;
		LOG(INFO) << i << ": mean = " << mean << " std = " << sqrt(mean_2 - mean * mean) << " num = " << bbox_target_num_;
	}

	all_rois_num = 0;
	for (int batch_id = 0; batch_id < batch_size; ++batch_id)
	{
		all_rois_num += batch_rois_num[batch_id];
	}

	vector<int> top_shape(2);
	top_shape[0] = all_rois_num;
	top_shape[1] = 7;
	top[0]->Reshape(top_shape);
	top_shape[0] = all_rois_num;
	top_shape[1] = 1;
	top[1]->Reshape(top_shape);
	top_shape[0] = all_rois_num;
	top_shape[1] = class_num * 6;
	top[2]->Reshape(top_shape);
	top[3]->Reshape(top_shape);
	top[4]->Reshape(top_shape);

	Dtype* top0_data = top[0]->mutable_cpu_data();
	Dtype* top1_data = top[1]->mutable_cpu_data();
	Dtype* top2_data = top[2]->mutable_cpu_data();
	Dtype* top3_data = top[3]->mutable_cpu_data();
	Dtype* top4_data = top[4]->mutable_cpu_data();
	int top0_offset = 0;
	int top1_offset = 0;
	int top2_offset = 0;
	int top3_offset = 0;
	int top4_offset = 0;
	for (int batch_id = 0; batch_id < batch_size; ++batch_id)
	{
		memcpy(top0_data + top0_offset, sample_rois[batch_id], sizeof(Dtype) * batch_rois_num[batch_id] * 7);
		memcpy(top1_data + top1_offset, sample_labels[batch_id], sizeof(Dtype) * batch_rois_num[batch_id]);
		memcpy(top2_data + top2_offset, bbox_targets[batch_id], sizeof(Dtype) * batch_rois_num[batch_id] * class_num * 6);
		memcpy(top3_data + top3_offset, bbox_inside_weights[batch_id], sizeof(Dtype) * batch_rois_num[batch_id] * class_num * 6);
		memcpy(top4_data + top4_offset, bbox_outside_weights[batch_id], sizeof(Dtype) * batch_rois_num[batch_id] * class_num * 6);
		top0_offset += batch_rois_num[batch_id] * 7;
		top1_offset += batch_rois_num[batch_id];
		top2_offset += batch_rois_num[batch_id] * class_num * 6;
		top3_offset += batch_rois_num[batch_id] * class_num * 6;
		top4_offset += batch_rois_num[batch_id] * class_num * 6;
	}

	for (int batch_id = 0; batch_id < batch_size; ++batch_id)
	{
		delete[]sample_rois[batch_id];
		delete[]sample_labels[batch_id];
		delete[]bbox_targets[batch_id];
		delete[]bbox_inside_weights[batch_id];
		delete[]bbox_outside_weights[batch_id];
	}
	delete[]sample_rois;
	delete[]sample_labels;
	delete[]bbox_targets;
	delete[]bbox_inside_weights;
	delete[]bbox_outside_weights;
	delete[]batch_rois_num;
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(ProposalTargetLayer);
REGISTER_LAYER_CLASS(ProposalTarget);

}  // namespace caffe