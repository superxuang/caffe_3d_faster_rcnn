#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/slice_predict_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <itkImage.h>
#include <itkMetaImageIOFactory.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

using std::sqrt;
using std::floor;
using std::ceil;
using std::min;
using std::max;
using std::exp;
using std::log;

namespace caffe {

template <typename Dtype>
void SlicePredictLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());
	typedef itk::ImageFileReader<ImageType> ImageReaderType;
	typedef itk::ImageFileReader<LabelType> LabelReaderType;

	const SlicePredictParameter& slice_predict_param = this->layer_param_.slice_predict_param();
	const unsigned int rng_seed = caffe_rng_rand();
	rng_.reset(new Caffe::RNG(rng_seed));
	const int contour_num_ = slice_predict_param.contour_name_list().name_size();
	std::ifstream infile(slice_predict_param.source().c_str());
	string line;
	size_t pos1, pos2;
	while (std::getline(infile, line)) {
		pos1 = line.find_first_of(' ');
		pos2 = line.find_last_of(' ');
		string image_file_name = line.substr(0, pos1);
		string label_file_name = line.substr(pos1 + 1, pos2 - pos1 - 1);
		string info_file_name = line.substr(pos2 + 1);
		std::ifstream infile_info(slice_predict_param.input_path() + info_file_name);
		std::vector<int> contour_labels;
		std::vector<int> exist_contours;
		while (std::getline(infile_info, line)) {
			pos1 = string::npos;
			for (int i = 0; i < contour_num_; ++i) {
				pos1 = line.find(slice_predict_param.contour_name_list().name(i));
				if (pos1 != string::npos) {
					exist_contours.push_back(i + 1);
					break;
				}
			}
			if (pos1 == string::npos)
				continue;

			pos1 = line.find_first_of(' ', pos1);
			pos2 = line.find_first_of(' ', pos1 + 1);
			int label_value = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
			contour_labels.push_back(label_value);
		}
		if (!contour_labels.empty()) {
			ImageReaderType::Pointer reader_image = ImageReaderType::New();
			reader_image->SetFileName(slice_predict_param.input_path() + image_file_name);
			reader_image->UpdateOutputInformation();
			ImageType::Pointer image = reader_image->GetOutput();
			const ImageType::SizeType& image_size = image->GetLargestPossibleRegion().GetSize();
			const ImageType::SpacingType& image_spacing = image->GetSpacing();

			size_t pos = image_file_name.find_first_of('/');
			image_file_name = image_file_name.substr(pos + 1);
			for (int i = 0; i < 3; ++i)
			{
				for (int j = 0; j < image_size[2 - i]; ++j)
				{
					lines_.push_back(std::make_pair(image_file_name, std::make_pair(i, j)));
				}
			}
		}
	}
	lines_id_ = 0;
}

template <typename Dtype>
void SlicePredictLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
  // Reshaping happens during the call to forward.
}

template <typename Dtype>
void SlicePredictLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const SlicePredictParameter& slice_predict_param = this->layer_param_.slice_predict_param();
	const int batch_size = bottom[0]->shape(0);
	CHECK_EQ(bottom[0]->shape(1), 2);
	const int class_num = bottom[0]->shape(2);
	const Dtype* batch_data = bottom[0]->cpu_data();
	Dtype* label_data = NULL;
	bool output_label = bottom.size() > 1;
	if (output_label)
	{
		label_data = bottom[1]->mutable_cpu_data();
	}
	const int line_size = lines_.size();
	for (int batch_ind = 0; batch_ind < batch_size; ++batch_ind)
	{
		if (lines_id_ + batch_ind > line_size - 1)
		{
			break;
		}
		const string& image_file_name = lines_[lines_id_ + batch_ind].first;
		const int& direction = lines_[lines_id_ + batch_ind].second.first;
		const int& slice_ind = lines_[lines_id_ + batch_ind].second.second;
		ofstream file(slice_predict_param.output_path() + image_file_name + ".slice.txt", ios::out | ios::app);
		if (file.is_open())
		{
			if (output_label)
			{
				file << direction << " " << slice_ind << " ";
				for (int i = 0; i < class_num; ++i)
				{
					file /*<< batch_data[batch_ind * 2 * class_num + class_num * 1 + i] << " " */<< label_data[batch_ind * class_num + i] << " ";
				}
				file << std::endl;
			}
			else
			{
				file << direction << " " << slice_ind << " ";
				for (int i = 0; i < class_num; ++i)
				{
					file << batch_data[batch_ind * 2 * class_num + class_num * 1 + i] << " ";
				}
				file << std::endl;
			}
			file.close();
		}

		if (lines_id_ + batch_ind + 1 >= line_size || image_file_name != lines_[lines_id_ + batch_ind + 1].first)
		{
			std::string input_file_name = slice_predict_param.output_path() + image_file_name + ".slice.txt";
			std::ifstream input_file(input_file_name);
			string line;
			size_t pos1, pos2;
			std::vector<bool>* pred_label = new std::vector<bool>[class_num * 3];
			while (std::getline(input_file, line)) {
				pos1 = 0;
				pos2 = line.find_first_of(' ', pos1);
				int direction = atoi(line.substr(pos1, pos2 - pos1).c_str());

				pos1 = pos2;
				pos2 = line.find_first_of(' ', pos1 + 1);
				int slice_ind = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());

				for (int i = 0; i < class_num; ++i)
				{
					pos1 = pos2;
					pos2 = line.find_first_of(' ', pos1 + 1);
					double class_prob = atof(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
					pred_label[direction * class_num + i].push_back(class_prob > 0.5);
				}
			}
			input_file.close();
			bool* bbox_exist = new bool[class_num];
			for (int class_ind = 0; class_ind < class_num; ++class_ind)
			{
				bbox_exist[class_ind] = true;
			}
			int* bbox = new int[class_num * 6];
			
			for (int direction = 0; direction < 3; ++direction)
			{
				for (int class_ind = 0; class_ind < class_num; ++class_ind)
				{
					const int slice_num = pred_label[direction * class_num + class_ind].size();
					std::vector<int> start_inds;
					std::vector<int> end_inds;
					for (int slice_ind = 0; slice_ind < slice_num; ++slice_ind)
					{
						if (pred_label[direction * class_num + class_ind][slice_ind]) 
						{
							if (slice_ind == 0 || !pred_label[direction * class_num + class_ind][slice_ind - 1])
							{
								start_inds.push_back(slice_ind);
							}
							if (slice_ind == slice_num - 1 || !pred_label[direction * class_num + class_ind][slice_ind + 1])
							{
								end_inds.push_back(slice_ind);
							}
						}
					}
					int max_length = 0;
					int start_ind = 0;
					int end_ind = 0;
					for (int i = 0; i < start_inds.size(); ++i)
					{
						if (end_inds[i] - start_inds[i] + 1 > max_length)
						{
							max_length = end_inds[i] - start_inds[i] + 1;
							start_ind = start_inds[i];
							end_ind = end_inds[i];
						}
					}
					if (max_length == 0)
					{
						bbox_exist[class_ind] = false;
					}
					bbox[class_ind * 6 + (2 - direction) + 0] = start_ind;
					bbox[class_ind * 6 + (2 - direction) + 3] = end_ind;
				}
			}
			remove(input_file_name.c_str());
			ofstream output_file(slice_predict_param.output_path() + image_file_name + ".pred.txt", ios::out | ios::trunc);
			if (output_file.is_open())
			{
				for (int i = 0; i < class_num; ++i)
				{
					if (bbox_exist[i])
					{
						output_file << i + 1 << " ";
						for (int j = 0; j < 6; ++j)
						{
							output_file << bbox[i * 6 + j] << " ";
						}
						output_file << 1 << std::endl;
					}
				}
				output_file.close();
			}
			delete[]bbox_exist;
			delete[]bbox;
		}
	}
	lines_id_ += batch_size;
}

template <typename Dtype>
void SlicePredictLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(SlicePredictLayer);
REGISTER_LAYER_CLASS(SlicePredict);

}  // namespace caffe