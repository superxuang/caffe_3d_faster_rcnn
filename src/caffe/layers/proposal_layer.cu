#include <iostream>  // NOLINT(readability/streams)
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>

#include "caffe/layers/proposal_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


using std::sqrt;
using std::floor;
using std::ceil;
using std::min;
using std::max;
using std::exp;
using std::log;

namespace caffe {

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	const ProposalParameter& proposal_param = this->layer_param_.proposal_param();

	const int batch_size = bottom[0]->shape(0);
	const int class_num = bottom[0]->shape(1) / anchors_num_;
	CHECK_EQ(class_num, 2) << "class_num != 2";
	const int proposal_num = proposal_param.proposal_num();

	feat_size_[0] = bottom[0]->shape(4);
	feat_size_[1] = bottom[0]->shape(3);
	feat_size_[2] = bottom[0]->shape(2);

	vector<int> top_shape(2);
	top_shape[0] = batch_size * proposal_num;
	top_shape[1] = 7;
	top[0]->Reshape(top_shape);

	// Enumerate all anchors
	const int feat_stride_xy = proposal_param.feat_stride_xy();
	const int feat_stride_z = proposal_param.feat_stride_z();
	const int all_anchors_num = feat_size_[2] * feat_size_[1] * feat_size_[0] * anchors_num_;
	Dtype* all_anchors = new Dtype[all_anchors_num * 6];
#pragma omp parallel for
	for (int shift_z = 0; shift_z < feat_size_[2]; ++shift_z) {
		for (int shift_y = 0; shift_y < feat_size_[1]; ++shift_y) {
			for (int shift_x = 0; shift_x < feat_size_[0]; ++shift_x) {
				for (int m = 0; m < anchors_num_; ++m) {
					all_anchors[(((shift_z * feat_size_[1] + shift_y) * feat_size_[0] + shift_x) * anchors_num_ + m) * 6 + 0] =
						shift_x * feat_stride_xy + anchors_[m * 6 + 0];
					all_anchors[(((shift_z * feat_size_[1] + shift_y) * feat_size_[0] + shift_x) * anchors_num_ + m) * 6 + 1] =
						shift_y * feat_stride_xy + anchors_[m * 6 + 1];
					all_anchors[(((shift_z * feat_size_[1] + shift_y) * feat_size_[0] + shift_x) * anchors_num_ + m) * 6 + 2] =
						shift_z * feat_stride_z + anchors_[m * 6 + 2];

					all_anchors[(((shift_z * feat_size_[1] + shift_y) * feat_size_[0] + shift_x) * anchors_num_ + m) * 6 + 3] =
						shift_x * feat_stride_xy + anchors_[m * 6 + 3];
					all_anchors[(((shift_z * feat_size_[1] + shift_y) * feat_size_[0] + shift_x) * anchors_num_ + m) * 6 + 4] =
						shift_y * feat_stride_xy + anchors_[m * 6 + 4];
					all_anchors[(((shift_z * feat_size_[1] + shift_y) * feat_size_[0] + shift_x) * anchors_num_ + m) * 6 + 5] =
						shift_z * feat_stride_z + anchors_[m * 6 + 5];
				}
			}
		}
	}

	for (int batch_id = 0; batch_id < batch_size; ++batch_id)
	{

		const Dtype* scores_src = bottom[0]->cpu_data() + batch_id * all_anchors_num * class_num + all_anchors_num;
		const Dtype* bbox_deltas_src = bottom[1]->cpu_data() + batch_id * all_anchors_num * 6;
		const Dtype* im_info = bottom[2]->cpu_data() + batch_id * 19;
		const int image_size[3] = { (int)im_info[2], (int)im_info[1], (int)im_info[0] };
		const double image_spacing[3] = { im_info[5], im_info[4], im_info[3] };
		const double image_origin[3] = { im_info[8], im_info[7], im_info[6] };
		//const int image_origin_size[3] = { (int)im_info[11], (int)im_info[10], (int)im_info[9] };
		const double image_origin_spacing[3] = { im_info[14], im_info[13], im_info[12] };
		const double image_origin_origin[3] = { im_info[17], im_info[16], im_info[15] };
		const int image_scale = im_info[18];
		//LOG(INFO) << "image size(L x H x W): " << im_info[0] << "x" << im_info[1] << "x" << im_info[2];
		//DLOG(INFO) << "scale: " << im_info[3];
		//LOG(INFO) << "score map size(L x H x W): " << feat_size_[2] << "x" << feat_size_[1] << "x" << feat_size_[0];

		Dtype* bbox_deltas = new Dtype[all_anchors_num * 6];
		Dtype* scores = new Dtype[all_anchors_num];
		//double score_max = -1000000;
		//double score_min = 1000000;
#pragma omp parallel for
		for (int m = 0; m < anchors_num_; ++m) {
			for (int l = 0; l < feat_size_[2]; ++l) {
				for (int h = 0; h < feat_size_[1]; ++h) {
					for (int w = 0; w < feat_size_[0]; ++w) {
						for (int n = 0; n < 6; ++n) {
							bbox_deltas[(((l * feat_size_[1] + h) * feat_size_[0] + w) * anchors_num_ + m) * 6 + n] =
								bbox_deltas_src[(((m * 6 + n) * feat_size_[2] + l) * feat_size_[1] + h) * feat_size_[0] + w];
						}
						scores[((l * feat_size_[1] + h) * feat_size_[0] + w) * anchors_num_ + m] =
							scores_src[((m * feat_size_[2] + l) * feat_size_[1] + h) * feat_size_[0] + w];
						//if (score_min > scores[((l * feat_size_[1] + h) * feat_size_[0] + w) * anchors_num_ + m]) {
						//	score_min = scores[((l * feat_size_[1] + h) * feat_size_[0] + w) * anchors_num_ + m];
						//}
						//if (score_max < scores[((l * feat_size_[1] + h) * feat_size_[0] + w) * anchors_num_ + m]) {
						//	score_max = scores[((l * feat_size_[1] + h) * feat_size_[0] + w) * anchors_num_ + m];
						//}
					}
				}
			}
		}
		//LOG(INFO) << "max score = " << score_max;
		//LOG(INFO) << "min score = " << score_min;

		Dtype* proposal = new Dtype[all_anchors_num * 6];
		bool* proposal_keep = new bool[all_anchors_num];
		int keep_num = 0;
		double min_size = proposal_param.rpn_min_size() * image_scale;
//#pragma omp parallel for
		for (int i = 0; i < all_anchors_num; ++i) {
			// Convert anchors into proposals via bbox transformations
			double width = all_anchors[i * 6 + 3] - all_anchors[i * 6 + 0] + 1.0;
			double height = all_anchors[i * 6 + 4] - all_anchors[i * 6 + 1] + 1.0;
			double length = all_anchors[i * 6 + 5] - all_anchors[i * 6 + 2] + 1.0;
			double ctr_x = all_anchors[i * 6 + 0] + 0.5 * width;
			double ctr_y = all_anchors[i * 6 + 1] + 0.5 * height;
			double ctr_z = all_anchors[i * 6 + 2] + 0.5 * length;
			if (proposal_param.bbox_normalize())
			{
				for (int j = 0; j < 6; ++j)
				{
					bbox_deltas[i * 6 + j] = bbox_deltas[i * 6 + j] * proposal_param.bbox_normalize_std(j) + proposal_param.bbox_normalize_mean(j);
				}
			}
			double proposal_ctr_x = bbox_deltas[i * 6 + 0] * width + ctr_x;
			double proposal_ctr_y = bbox_deltas[i * 6 + 1] * height + ctr_y;
			double proposal_ctr_z = bbox_deltas[i * 6 + 2] * length + ctr_z;
			double proposal_w = exp(bbox_deltas[i * 6 + 3]) * width;
			double proposal_h = exp(bbox_deltas[i * 6 + 4]) * height;
			double proposal_l = exp(bbox_deltas[i * 6 + 5]) * length;
			proposal[i * 6 + 0] = proposal_ctr_x - 0.5 * proposal_w;
			proposal[i * 6 + 1] = proposal_ctr_y - 0.5 * proposal_h;
			proposal[i * 6 + 2] = proposal_ctr_z - 0.5 * proposal_l;
			proposal[i * 6 + 3] = proposal_ctr_x + 0.5 * proposal_w;
			proposal[i * 6 + 4] = proposal_ctr_y + 0.5 * proposal_h;
			proposal[i * 6 + 5] = proposal_ctr_z + 0.5 * proposal_l;

			// clip predicted boxes to image
			proposal[i * 6 + 0] = max(min(static_cast<double>(proposal[i * 6 + 0]), image_size[0] - 1.0), 0.0);
			proposal[i * 6 + 1] = max(min(static_cast<double>(proposal[i * 6 + 1]), image_size[1] - 1.0), 0.0);
			proposal[i * 6 + 2] = max(min(static_cast<double>(proposal[i * 6 + 2]), image_size[2] - 1.0), 0.0);
			proposal[i * 6 + 3] = max(min(static_cast<double>(proposal[i * 6 + 3]), image_size[0] - 1.0), 0.0);
			proposal[i * 6 + 4] = max(min(static_cast<double>(proposal[i * 6 + 4]), image_size[1] - 1.0), 0.0);
			proposal[i * 6 + 5] = max(min(static_cast<double>(proposal[i * 6 + 5]), image_size[2] - 1.0), 0.0);

			proposal_keep[i] = true;

			// only keep anchors inside the image
			proposal_keep[i] &= (
				all_anchors[i * 6 + 0] >= 0 &&
				all_anchors[i * 6 + 1] >= 0 &&
				all_anchors[i * 6 + 2] >= 0 &&
				all_anchors[i * 6 + 3] < im_info[2] &&
				all_anchors[i * 6 + 4] < im_info[1] &&
				all_anchors[i * 6 + 5] < im_info[0]);

			// remove predicted boxes with either height or width < threshold
			// (NOTE: convert rpn_min_size to input image scale stored in im_info[3])
			proposal_keep[i] &= (
				(proposal[i * 6 + 3] - proposal[i * 6 + 0] + 1) >= min_size &&
				(proposal[i * 6 + 4] - proposal[i * 6 + 1] + 1) >= min_size &&
				(proposal[i * 6 + 5] - proposal[i * 6 + 2] + 1) >= min_size);
			keep_num += proposal_keep[i];
		}
		delete[]bbox_deltas;

		Dtype* scores_tmp = new Dtype[keep_num];
		Dtype* proposal_tmp = new Dtype[keep_num * 6];
		keep_num = 0;
		for (int i = 0; i < all_anchors_num; ++i) {
			if (proposal_keep[i]) {
				scores_tmp[keep_num] = scores[i];
				for (int j = 0; j < 6; ++j) {
					proposal_tmp[keep_num * 6 + j] = proposal[i * 6 + j];
				}
				keep_num++;
			}
		}
		delete[]proposal_keep;
		delete[]scores;
		delete[]proposal;
		scores = scores_tmp;
		proposal = proposal_tmp;

		// sort all (proposal, score) pairs by score from highest to lowest
		// take top pre_nms_topN (e.g. 6000)
		int pre_nms_top_n = proposal_param.rpn_pre_nms_top_n();
		int pre_nms_num = min(keep_num, pre_nms_top_n);
		Dtype tmp;
		for (int i = keep_num - 1; i >= keep_num - pre_nms_num; --i) {
			for (int j = keep_num - 1; j >= keep_num - i; --j) {
				if (scores[j] > scores[j - 1]) {
					tmp = scores[j - 1];
					scores[j - 1] = scores[j];
					scores[j] = tmp;
					for (int k = 0; k < 6; ++k) {
						tmp = proposal[(j - 1) * 6 + k];
						proposal[(j - 1) * 6 + k] = proposal[j * 6 + k];
						proposal[j * 6 + k] = tmp;
					}
				}
			}
		}

		// nms
		const int post_nms_top_n = proposal_param.rpn_post_nms_top_n();
		const int post_nms_num = min(pre_nms_num, post_nms_top_n);
		const double nms_thresh = proposal_param.rpn_nms_threshold();
		bool* suppressed = new bool[pre_nms_num];
		int* nms_keep_indices = new int[pre_nms_num];
		int nms_keep_num = 0;
		memset(suppressed, 0, pre_nms_num * sizeof(bool));
		for (int i = 0; i < pre_nms_num; ++i) {
			if (suppressed[i]) {
				continue;
			}
			nms_keep_indices[nms_keep_num++] = i;
			if (nms_keep_num >= post_nms_num) {
				break;
			}
			double ix1 = proposal[i * 6 + 0];
			double iy1 = proposal[i * 6 + 1];
			double iz1 = proposal[i * 6 + 2];
			double ix2 = proposal[i * 6 + 3];
			double iy2 = proposal[i * 6 + 4];
			double iz2 = proposal[i * 6 + 5];
			double volume_i = (ix2 - ix1 + 1) * (iy2 - iy1 + 1) * (iz2 - iz1 + 1);
			for (int j = i + 1; j < pre_nms_num; ++j) {
				if (suppressed[j]) {
					continue;
				}
				double jx1 = proposal[j * 6 + 0];
				double jy1 = proposal[j * 6 + 1];
				double jz1 = proposal[j * 6 + 2];
				double jx2 = proposal[j * 6 + 3];
				double jy2 = proposal[j * 6 + 4];
				double jz2 = proposal[j * 6 + 5];
				double volume_j = (jx2 - jx1 + 1) * (jy2 - jy1 + 1) * (jz2 - jz1 + 1);
				double xx1 = max(ix1, jx1);
				double yy1 = max(iy1, jy1);
				double zz1 = max(iz1, jz1);
				double xx2 = min(ix2, jx2);
				double yy2 = min(iy2, jy2);
				double zz2 = min(iz2, jz2);
				double w = max(0.0, xx2 - xx1 + 1);
				double h = max(0.0, yy2 - yy1 + 1);
				double l = max(0.0, zz2 - zz1 + 1);
				double inter = w * h * l;
				double overlap = inter / (volume_i + volume_j - inter);
				if (overlap >= nms_thresh) {
					suppressed[j] = true;
				}
			}
		}
		delete[]suppressed;

		int bg_num = 0;
		Dtype* bg_scores = NULL;
		Dtype* bg_proposals = NULL;
		if (this->phase() == TRAIN) {
			int bg_num_pre = 0;
			Dtype* bg_scores_pre = new Dtype[keep_num];
			Dtype* bg_proposals_pre = new Dtype[keep_num * 6];
			for (int i = pre_nms_num; i < keep_num; ++i) {
				if (scores[i] < proposal_param.bg_score_threshold()) {
					bg_scores_pre[bg_num_pre] = scores[i];
					for (int j = 0; j < 6; ++j) {
						bg_proposals_pre[bg_num_pre * 6 + j] = proposal[i * 6 + j];
					}
					bg_num_pre++;
				}
			}
			if (bg_num_pre == 0) {
				for (int i = pre_nms_num; i < keep_num; ++i) {
					bg_scores_pre[bg_num_pre] = scores[i];
					for (int j = 0; j < 6; ++j) {
						bg_proposals_pre[bg_num_pre * 6 + j] = proposal[i * 6 + j];
					}
					bg_num_pre++;
				}
			}
			if (bg_num_pre > 0) {
				if (bg_num_pre <= proposal_num - nms_keep_num)
				{
					bg_num = bg_num_pre;
					bg_scores = new Dtype[bg_num_pre];
					bg_proposals = new Dtype[bg_num_pre * 6];
					for (int i = 0; i < bg_num_pre; ++i) {
						bg_scores[i] = bg_scores_pre[i];
						for (int j = 0; j < 6; ++j) {
							bg_proposals[i * 6 + j] = bg_proposals_pre[i * 6 + j];
						}
					}
				}
				else
				{
					int* bg_inds = new int[proposal_num - nms_keep_num];
					bg_scores = new Dtype[proposal_num - nms_keep_num];
					bg_proposals = new Dtype[(proposal_num - nms_keep_num) * 6];
					caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
					while (bg_num < proposal_num - nms_keep_num && bg_num < bg_num_pre) {
						int i = (*rng)() % bg_num_pre;
						bool exists = false;
						for (int j = 0; j < bg_num; ++j) {
							if (bg_inds[j] == i) {
								exists = true;
								break;
							}
						}
						bg_inds[bg_num] = i;
						if (!exists) {
							bg_scores[bg_num] = bg_scores_pre[i];
							for (int j = 0; j < 6; ++j) {
								bg_proposals[bg_num * 6 + j] = bg_proposals_pre[i * 6 + j];
							}
							bg_num++;
						}
					}
					delete[]bg_inds;
				}
			}
			delete[]bg_scores_pre;
			delete[]bg_proposals_pre;
		}

		//FILE* output_file;
		//output_file = fopen("F:/proposal.txt", "w");
		//if (output_file != NULL)
		{
			Dtype* roi_data = top[0]->mutable_cpu_data() + batch_id * proposal_num * 7;
			for (int i = 0; i < proposal_num; ++i)
			{
				roi_data[i * 7 + 0] = -1;
			}
			for (int i = 0; i < nms_keep_num; ++i) {
				roi_data[i * 7 + 0] = batch_id;
				for (int j = 0; j < 6; ++j) {
					roi_data[i * 7 + 1 + j] = proposal[nms_keep_indices[i] * 6 + j];
				}
				//fprintf(output_file, "%d %f %f %f %f %f %f %f\n",
				//	1,					
				//	(roi_data[i * 7 + 1] * image_spacing[0] + image_origin[0] - image_origin_origin[0]) / image_origin_spacing[0],
				//	(roi_data[i * 7 + 2] * image_spacing[1] + image_origin[1] - image_origin_origin[1]) / image_origin_spacing[1],
				//	(roi_data[i * 7 + 3] * image_spacing[2] + image_origin[2] - image_origin_origin[2]) / image_origin_spacing[2],
				//	(roi_data[i * 7 + 4] * image_spacing[0] + image_origin[0] - image_origin_origin[0]) / image_origin_spacing[0],
				//	(roi_data[i * 7 + 5] * image_spacing[1] + image_origin[1] - image_origin_origin[1]) / image_origin_spacing[1],
				//	(roi_data[i * 7 + 6] * image_spacing[2] + image_origin[2] - image_origin_origin[2]) / image_origin_spacing[2],
				//	1);
			}
			for (int i = 0; i < bg_num; ++i) {
				roi_data[(i + nms_keep_num) * 7 + 0] = batch_id;
				for (int j = 0; j < 6; ++j) {
					roi_data[(i + nms_keep_num) * 7 + 1 + j] = bg_proposals[i * 6 + j];
				}
				//fprintf(output_file, "%d %f %f %f %f %f %f %f\n",
				//	1,
				//	(roi_data[(i + nms_keep_num) * 7 + 1] * image_spacing[0] + image_origin[0] - image_origin_origin[0]) / image_origin_spacing[0],
				//	(roi_data[(i + nms_keep_num) * 7 + 2] * image_spacing[1] + image_origin[1] - image_origin_origin[1]) / image_origin_spacing[1],
				//	(roi_data[(i + nms_keep_num) * 7 + 3] * image_spacing[2] + image_origin[2] - image_origin_origin[2]) / image_origin_spacing[2],
				//	(roi_data[(i + nms_keep_num) * 7 + 4] * image_spacing[0] + image_origin[0] - image_origin_origin[0]) / image_origin_spacing[0],
				//	(roi_data[(i + nms_keep_num) * 7 + 5] * image_spacing[1] + image_origin[1] - image_origin_origin[1]) / image_origin_spacing[1],
				//	(roi_data[(i + nms_keep_num) * 7 + 6] * image_spacing[2] + image_origin[2] - image_origin_origin[2]) / image_origin_spacing[2],
				//	1);
			}
		}
		//fclose(output_file);

		delete[]nms_keep_indices;
		delete[]scores;
		delete[]proposal;
		delete[]bg_scores;
		delete[]bg_proposals;
	}
	delete[]all_anchors;
}

template <typename Dtype>
void ProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(ProposalLayer);

}  // namespace caffe