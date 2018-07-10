#include <iostream>  // NOLINT(readability/streams)
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>

#include "caffe/layers/rpn_output_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

//#define OUTPUT_BOX_TO_FILE

using std::sqrt;
using std::floor;
using std::ceil;
using std::min;
using std::max;
using std::exp;
using std::log;

namespace caffe {

template <typename Dtype>
void RPNOutputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	const int batch_size = bottom[0]->shape(0);
	const int class_num = bottom[0]->shape(1) / anchors_num_;
	const int shift_offset[3][7] = {
		0, -5, 5, 0, 0, 0, 0,
		0, 0, 0, -5, 5, 0, 0,
		0, 0, 0, 0, 0, -1, 1
	};
	//LOG(INFO) << "input image size(L x H x W): " << im_info[0] << "x" << im_info[1] << "x" << im_info[2];
	feat_size_[0] = bottom[0]->shape(4);
	feat_size_[1] = bottom[0]->shape(3);
	feat_size_[2] = bottom[0]->shape(2);
	//LOG(INFO) << "score map size(L x H x W): " << feat_size_[2] << "x" << feat_size_[1] << "x" << feat_size_[0];

	const RPNOutputParameter& rpn_output_param = this->layer_param_.rpn_output_param();
	const int feat_stride_xy = rpn_output_param.feat_stride_xy();
	const int feat_stride_z = rpn_output_param.feat_stride_z();
	const int all_anchors_num = feat_size_[2] * feat_size_[1] * feat_size_[0] * anchors_num_;
	double* pred_box = new double[batch_size * class_num * 7];
	memset(pred_box, 0, sizeof(double) * batch_size * class_num * 7);

	// generate all ref_boxes
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

	for (int batch_id = 0; batch_id < batch_size; ++batch_id) {

		const Dtype* im_info = bottom[2]->cpu_data() + 19 * batch_id;
		const int image_size[3] = { (int)im_info[2], (int)im_info[1], (int)im_info[0] };
		const double image_spacing[3] = { im_info[5], im_info[4], im_info[3] };
		const double image_origin[3] = { im_info[8], im_info[7], im_info[6] };
		//const int image_origin_size[3] = { (int)im_info[11], (int)im_info[10], (int)im_info[9] };
		const double image_origin_spacing[3] = { im_info[14], im_info[13], im_info[12] };
		const double image_origin_origin[3] = { im_info[17], im_info[16], im_info[15] };
		const int image_scale = im_info[18];

		const Dtype* bbox_deltas_src = bottom[1]->cpu_data() + batch_id * all_anchors_num * 6;

		Dtype* bbox_deltas = new Dtype[all_anchors_num * 6];
		for (int m = 0; m < anchors_num_; ++m) {
			for (int l = 0; l < feat_size_[2]; ++l) {
				for (int h = 0; h < feat_size_[1]; ++h) {
					for (int w = 0; w < feat_size_[0]; ++w) {
						for (int n = 0; n < 6; ++n) {
							bbox_deltas[(((l * feat_size_[1] + h) * feat_size_[0] + w) * anchors_num_ + m) * 6 + n] =
								bbox_deltas_src[(((m * 6 + n) * feat_size_[2] + l) * feat_size_[1] + h) * feat_size_[0] + w];
						}
					}
				}
			}
		}
		Dtype* proposal = new Dtype[all_anchors_num * 6];
		bool* keep_proposal = new bool[all_anchors_num];
		int keep_num = 0;
		double min_size = rpn_output_param.rpn_min_size() * image_scale;
//#pragma omp parallel for
		for (int i = 0; i < all_anchors_num; ++i) {
			// Convert anchors into proposals via bbox transformations
			double width = all_anchors[i * 6 + 3] - all_anchors[i * 6 + 0] + 1.0;
			double height = all_anchors[i * 6 + 4] - all_anchors[i * 6 + 1] + 1.0;
			double length = all_anchors[i * 6 + 5] - all_anchors[i * 6 + 2] + 1.0;
			double ctr_x = all_anchors[i * 6 + 0] + 0.5 * width;
			double ctr_y = all_anchors[i * 6 + 1] + 0.5 * height;
			double ctr_z = all_anchors[i * 6 + 2] + 0.5 * length;
			if (rpn_output_param.bbox_normalize())
			{
				for (int j = 0; j < 6; ++j)
				{
					bbox_deltas[i * 6 + j] = bbox_deltas[i * 6 + j] * rpn_output_param.bbox_normalize_std(j) + rpn_output_param.bbox_normalize_mean(j);
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

			keep_proposal[i] = true;

			// only keep anchors inside the image
			keep_proposal[i] &= (
				all_anchors[i * 6 + 0] >= 0 &&
				all_anchors[i * 6 + 1] >= 0 &&
				all_anchors[i * 6 + 2] >= 0 &&
				all_anchors[i * 6 + 3] < im_info[2] &&
				all_anchors[i * 6 + 4] < im_info[1] &&
				all_anchors[i * 6 + 5] < im_info[0]);

			// remove predicted boxes with either height or width < threshold
			// (NOTE: convert rpn_min_size to input image scale stored in im_info[3])
			keep_proposal[i] &= (
				(proposal[i * 6 + 3] - proposal[i * 6 + 0] + 1) >= min_size &&
				(proposal[i * 6 + 4] - proposal[i * 6 + 1] + 1) >= min_size &&
				(proposal[i * 6 + 5] - proposal[i * 6 + 2] + 1) >= min_size);
			keep_num += keep_proposal[i];
		}
		delete[]bbox_deltas;

#ifdef OUTPUT_BOX_TO_FILE
		size_t pos = lines_[lines_id_].first.find_first_of('/');
		std::string file_name = lines_[lines_id_].first.substr(pos + 1);
		std::string output_file_name = rpn_output_param.roi_root_folder() + file_name + ".proposal.txt";
		FILE* file;
		file = fopen(output_file_name.c_str(), "w");
		if (file != NULL)
		{
#endif
			const Dtype* bg_score = bottom[0]->cpu_data() + batch_id * all_anchors_num * class_num;
			int* max_score_label = new int[all_anchors_num];
			memset(max_score_label, 0, sizeof(int) * all_anchors_num);
			Dtype* max_score = new Dtype[all_anchors_num];
			for (int m = 0; m < anchors_num_; ++m)
			{
				for (int l = 0; l < feat_size_[2]; ++l)
				{
					for (int h = 0; h < feat_size_[1]; ++h)
					{
						for (int w = 0; w < feat_size_[0]; ++w)
						{
							max_score[((l * feat_size_[1] + h) * feat_size_[0] + w) * anchors_num_ + m] =
								bg_score[((m * feat_size_[2] + l) * feat_size_[1] + h) * feat_size_[0] + w];
						}
					}
				}
			}
			for (int class_label = 1; class_label <= class_num - 1; ++class_label)
			{
				const Dtype* class_scores = bg_score + all_anchors_num * class_label;
				Dtype* score_buffer = new Dtype[all_anchors_num];
				for (int m = 0; m < anchors_num_; ++m)
				{
					for (int l = 0; l < feat_size_[2]; ++l)
					{
						for (int h = 0; h < feat_size_[1]; ++h)
						{
							for (int w = 0; w < feat_size_[0]; ++w)
							{
								score_buffer[((l * feat_size_[1] + h) * feat_size_[0] + w) * anchors_num_ + m] =
									class_scores[((m * feat_size_[2] + l) * feat_size_[1] + h) * feat_size_[0] + w];
							}
						}
					}
				}
				for (int i = 0; i < all_anchors_num; ++i)
				{
					if (keep_proposal[i]) {
						if (max_score[i] < score_buffer[i])
						{
							max_score[i] = score_buffer[i];
							max_score_label[i] = class_label;
						}
					}
				}
				delete[]score_buffer;
			}
			int* instance_num = new int[class_num];
			for (int i = 0; i < class_num; ++i)
			{
				instance_num[i] = 0;
			}
			for (int i = 0; i < all_anchors_num; ++i)
			{
				instance_num[max_score_label[i]]++;
			}
#ifdef OUTPUT_BOX_TO_FILE
			std::string all_box_file_name = rpn_output_param.roi_root_folder() + file_name + ".all.box.txt";
			FILE* all_box_file;
			all_box_file = fopen(all_box_file_name.c_str(), "w");
			if (all_box_file != NULL)
			{
#endif
				for (int class_label = 1; class_label < class_num; ++class_label)
				{
					int n = rpn_output_param.top_percent_proposal() * instance_num[class_label];
					if (n <= 0)
						n = 1;
					int* top_n_inds = new int[n];
					double* top_n_scores = new double[n];
					for (int i = 0; i < n; ++i)
					{
						top_n_inds[i] = -1;
						top_n_scores[i] = -1.0;
					}
					for (int i = 0; i < all_anchors_num; ++i)
					{
						if (max_score_label[i] == class_label)
						{
#ifdef OUTPUT_BOX_TO_FILE
							Dtype output_proposal[6];
							for (int j = 0; j < 2; ++j)
							{
								for (int k = 0; k < 3; ++k)
								{
									output_proposal[j * 3 + k] = ((proposal[i * 6 + j * 3 + k] + shift_offset[k][batch_id]) * image_spacing[k] + image_origin[k] - image_origin_origin[k]) / image_origin_spacing[k];
								}
							}

							fprintf(all_box_file, "%d %f %f %f %f %f %f %f\n",
								max_score_label[i],
								output_proposal[0],
								output_proposal[1],
								output_proposal[2],
								output_proposal[3],
								output_proposal[4],
								output_proposal[5],
								max_score[i]);
#endif

							if (max_score[i] > top_n_scores[0] && max_score[i] > rpn_output_param.fg_score_threshold())
							{
								top_n_inds[0] = i;
								top_n_scores[0] = max_score[i];
								for (int j = 0; j < n - 1; ++j)
								{
									if (top_n_scores[j] > top_n_scores[j + 1])
									{
										int tmp_ind = top_n_inds[j];
										top_n_inds[j] = top_n_inds[j + 1];
										top_n_inds[j + 1] = tmp_ind;
										double tmp_score = top_n_scores[j];
										top_n_scores[j] = top_n_scores[j + 1];
										top_n_scores[j + 1] = tmp_score;
									}
									else
									{
										break;
									}
								}
							}
						}
					}
					int post_n = n;
					for (int i = 0; i < n; ++i)
					{
						if (top_n_inds[i] < 0)
						{
							post_n--;
						}
						else
						{
							break;
						}
					}
					if (post_n > 0)
					{
						double avg_proposal[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
						if (rpn_output_param.weighted_top_n_proposal())
						{
							double weight_sum = 0.0;
							for (int i = n - 1; i >= n - post_n; --i)
							{
								weight_sum += top_n_scores[i];
								for (int j = 0; j < 6; ++j)
								{
									avg_proposal[j] += top_n_scores[i] * proposal[top_n_inds[i] * 6 + j];
								}
							}
							for (int i = 0; i < 6; ++i)
							{
								avg_proposal[i] = avg_proposal[i] / weight_sum;
							}
						}
						else
						{
							for (int i = n - 1; i >= n - post_n; --i)
							{
								for (int j = 0; j < 6; ++j)
								{
									avg_proposal[j] += proposal[top_n_inds[i] * 6 + j];
								}
							}
							for (int i = 0; i < 6; ++i)
							{
								avg_proposal[i] = avg_proposal[i] / (double)post_n;
							}
						}
						for (int i = 0; i < 2; ++i)
						{
							for (int j = 0; j < 3; ++j)
							{
								avg_proposal[i * 3 + j] = ((avg_proposal[i * 3 + j] + shift_offset[j][batch_id]) * image_spacing[j] + image_origin[j] - image_origin_origin[j]) / image_origin_spacing[j];
							}
						}
#ifdef OUTPUT_BOX_TO_FILE
						fprintf(file, "%d %f %f %f %f %f %f %f\n",
							class_label,
							avg_proposal[0],
							avg_proposal[1],
							avg_proposal[2],
							avg_proposal[3],
							avg_proposal[4],
							avg_proposal[5],
							top_n_scores[n - 1]);
#endif
						pred_box[batch_id * class_num * 7 + class_label * 7 + 0] = avg_proposal[0];
						pred_box[batch_id * class_num * 7 + class_label * 7 + 1] = avg_proposal[1];
						pred_box[batch_id * class_num * 7 + class_label * 7 + 2] = avg_proposal[2];
						pred_box[batch_id * class_num * 7 + class_label * 7 + 3] = avg_proposal[3];
						pred_box[batch_id * class_num * 7 + class_label * 7 + 4] = avg_proposal[4];
						pred_box[batch_id * class_num * 7 + class_label * 7 + 5] = avg_proposal[5];
						pred_box[batch_id * class_num * 7 + class_label * 7 + 6] = top_n_scores[n - 1];
					}
					delete[]top_n_inds;
					delete[]top_n_scores;
				}
#ifdef OUTPUT_BOX_TO_FILE
			}
			fclose(all_box_file);
#endif
			delete[]instance_num;
			delete[]max_score;
			delete[]max_score_label;
#ifdef OUTPUT_BOX_TO_FILE
		}
		fclose(file);
#endif
		delete[]proposal;
		delete[]keep_proposal;
	}

	delete[]all_anchors;

	size_t pos = lines_[lines_id_].first.find_first_of('/');
	std::string file_name = lines_[lines_id_].first.substr(pos + 1);
	std::string output_file_name = rpn_output_param.roi_root_folder() + file_name + ".pred.txt";
	FILE* output_file;
	output_file = fopen(output_file_name.c_str(), "w");
	if (output_file != NULL)
	{
		for (int class_id = 1; class_id < class_num; ++class_id)
		{
			int box_num = 0;
			double output_box[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			for (int batch_id = 0; batch_id < batch_size; ++batch_id)
			{
				if (pred_box[batch_id * class_num * 7 + class_id * 7 + 6] > 0)
				{
					box_num++;
					for (int i = 0; i < 7; ++i)
					{
						output_box[i] += pred_box[batch_id * class_num * 7 + class_id * 7 + i];
					}
				}
			}
			if (box_num > 0)
			{
				for (int i = 0; i < 7; ++i)
				{
					output_box[i] = output_box[i] / box_num;
				}
				fprintf(output_file, "%d %f %f %f %f %f %f %f\n",
					class_id,
					output_box[0],
					output_box[1],
					output_box[2],
					output_box[3],
					output_box[4],
					output_box[5],
					output_box[6]);
			}
		}
	}
	fclose(output_file);

	delete[]pred_box;

	lines_id_ = (lines_id_ + 1) % lines_.size();
}

template <typename Dtype>
void RPNOutputLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(RPNOutputLayer);

}  // namespace caffe