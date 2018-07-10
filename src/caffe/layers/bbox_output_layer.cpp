#include <algorithm>
#include <vector>

#include "caffe/layers/bbox_output_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BBoxOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (this->phase() == TEST) {
    const BBoxOutputParameter& bbox_output_param = this->layer_param_.bbox_output_param();
    
    // Read the file with filenames and labels
    const string& source = bbox_output_param.source();
    
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    string line;
    size_t pos1, pos2;
    while (std::getline(infile, line)) {
      pos1 = line.find_first_of(' ');
      pos2 = line.find_last_of(' ');
      string image_file_name = line.substr(0, pos1);
      string label_file_name = line.substr(pos1 + 1, pos2 - pos1 - 1);
      string info_file_name = line.substr(pos2 + 1);
      lines_.push_back(image_file_name);
    }
    
    CHECK(!lines_.empty()) << "File is empty";
    
    lines_id_ = 0;
  }
}

template <typename Dtype>
void BBoxOutputLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void BBoxOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const BBoxOutputParameter& bbox_output_param = this->layer_param_.bbox_output_param();
	const string& root_folder = bbox_output_param.root_folder();
	const string& output_root_folder = bbox_output_param.output_root_folder();
	const int roi_num = bottom[0]->num();
	const int class_num = bottom[1]->shape(1);
	const Dtype* box_data = bottom[0]->cpu_data();
	const Dtype* box_cls_data = bottom[1]->cpu_data();
	const Dtype* box_reg_data = bottom[2]->cpu_data();
	const Dtype* im_info = bottom[3]->cpu_data();
	const int image_size[3] = { (int)im_info[2], (int)im_info[1], (int)im_info[0] };
	const double image_spacing[3] = { im_info[5], im_info[4], im_info[3] };
	const double image_origin[3] = { im_info[8], im_info[7], im_info[6] };
	//const int image_origin_size[3] = { (int)im_info[11], (int)im_info[10], (int)im_info[9] };
	const double image_origin_spacing[3] = { im_info[14], im_info[13], im_info[12] };
	const double image_origin_origin[3] = { im_info[17], im_info[16], im_info[15] };
	const int image_scale = im_info[18];

	size_t pos = lines_[lines_id_].find_first_of('/');
	std::string file_name = lines_[lines_id_].substr(pos + 1);
	std::string output_file_name = bbox_output_param.output_root_folder() + file_name + ".pred.txt";

	FILE* file;
	file = fopen(output_file_name.c_str(), "w");
	if (file != NULL)
	{
		int* class_max_score_id = new int[class_num];
		double* class_max_score = new double[class_num];
		for (int i = 0; i < class_num; ++i) {
			class_max_score_id[i] = -1;
			class_max_score[i] = -1;
		}
		int* roi_label = new int[roi_num];
		double* roi_score = new double[roi_num];
		double* roi_bbox = new double[roi_num * 6];
		for (int i = 0; i < roi_num; ++i) {
			roi_label[i] = 0;
			roi_score[i] = box_cls_data[i * class_num + 0];
			for (int j = 1; j < class_num; ++j) {
				if (roi_score[i] < box_cls_data[i * class_num + j]) {
					roi_label[i] = j;
					roi_score[i] = box_cls_data[i * class_num + j];
				}
			}
			if (class_max_score[roi_label[i]] < roi_score[i]) {
				class_max_score_id[roi_label[i]] = i;
				class_max_score[roi_label[i]] = roi_score[i];
			}
			double w = box_data[i * 7 + 4] - box_data[i * 7 + 1] + 1;
			double h = box_data[i * 7 + 5] - box_data[i * 7 + 2] + 1;
			double l = box_data[i * 7 + 6] - box_data[i * 7 + 3] + 1;
			double ctr_x = box_data[i * 7 + 1] + 0.5 * w;
			double ctr_y = box_data[i * 7 + 2] + 0.5 * h;
			double ctr_z = box_data[i * 7 + 3] + 0.5 * l;
			double bbox_deltas[6] = {0, 0, 0, 0, 0, 0};
			if (bbox_output_param.apply_second_refinement())
			{
				if (bbox_output_param.bbox_normalize())
				{
					for (int j = 0; j < 6; ++j)
					{
						bbox_deltas[j] = box_reg_data[i * class_num * 6 + roi_label[i] * 6 + j] * bbox_output_param.bbox_normalize_std(j) + bbox_output_param.bbox_normalize_mean(j);
					}
				}
				else
				{
					for (int j = 0; j < 6; ++j)
					{
						bbox_deltas[j] = box_reg_data[i * class_num * 6 + roi_label[i] * 6 + j];
					}
				}
			}
			double roi_ctr_x = ctr_x + w * bbox_deltas[0];
			double roi_ctr_y = ctr_y + h * bbox_deltas[1];
			double roi_ctr_z = ctr_z + l * bbox_deltas[2];
			double roi_w = w * exp(bbox_deltas[3]);
			double roi_h = h * exp(bbox_deltas[4]);
			double roi_l = l * exp(bbox_deltas[5]);
			roi_bbox[i * 6 + 0] = roi_ctr_x - 0.5 * roi_w;
			roi_bbox[i * 6 + 1] = roi_ctr_y - 0.5 * roi_h;
			roi_bbox[i * 6 + 2] = roi_ctr_z - 0.5 * roi_l;
			roi_bbox[i * 6 + 3] = roi_ctr_x + 0.5 * roi_w;
			roi_bbox[i * 6 + 4] = roi_ctr_y + 0.5 * roi_h;
			roi_bbox[i * 6 + 5] = roi_ctr_z + 0.5 * roi_l;

			// clip predicted boxes to image
			roi_bbox[i * 6 + 0] = max(min(static_cast<double>(roi_bbox[i * 6 + 0]), image_size[0] - 1.0), 0.0);
			roi_bbox[i * 6 + 1] = max(min(static_cast<double>(roi_bbox[i * 6 + 1]), image_size[1] - 1.0), 0.0);
			roi_bbox[i * 6 + 2] = max(min(static_cast<double>(roi_bbox[i * 6 + 2]), image_size[2] - 1.0), 0.0);
			roi_bbox[i * 6 + 3] = max(min(static_cast<double>(roi_bbox[i * 6 + 3]), image_size[0] - 1.0), 0.0);
			roi_bbox[i * 6 + 4] = max(min(static_cast<double>(roi_bbox[i * 6 + 4]), image_size[1] - 1.0), 0.0);
			roi_bbox[i * 6 + 5] = max(min(static_cast<double>(roi_bbox[i * 6 + 5]), image_size[2] - 1.0), 0.0);

			for (int j = 0; j < 2; ++j)
			{
				for (int k = 0; k < 3; ++k)
				{
					roi_bbox[i * 6 + j * 3 + k] = (roi_bbox[i * 6 + j * 3 + k] * image_spacing[k] + image_origin[k] - image_origin_origin[k]) / image_origin_spacing[k];
				}
			}


		}

		double* final_roi = new double[class_num * 6];
		for (int i = 1; i < class_num; ++i)
		{
			if (class_max_score[i] > 0)
			{
				fprintf(file, "%d %f %f %f %f %f %f %f\n",
					i,
					roi_bbox[class_max_score_id[i] * 6 + 0],
					roi_bbox[class_max_score_id[i] * 6 + 1],
					roi_bbox[class_max_score_id[i] * 6 + 2],
					roi_bbox[class_max_score_id[i] * 6 + 3],
					roi_bbox[class_max_score_id[i] * 6 + 4],
					roi_bbox[class_max_score_id[i] * 6 + 5],
					class_max_score[i]);
			}
		}
	}
	fclose(file);

	lines_id_ = (lines_id_ + 1) % lines_.size();
}

template <typename Dtype>
void BBoxOutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(BBoxOutputLayer);
REGISTER_LAYER_CLASS(BBoxOutput);

}  // namespace caffe
