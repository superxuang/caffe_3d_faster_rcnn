#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/mhd_slice_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <itkMetaImageIOFactory.h>
#include <itkIdentityTransform.h>
#include <itkResampleImageFilter.h>
#include <itkBSplineTransform.h>
#include <itkBSplineTransformInitializer.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkHistogramMatchingImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkAffineTransform.h>

using std::floor;

namespace caffe {

	template <typename Dtype>
	MHDSliceDataLayer<Dtype>::~MHDSliceDataLayer<Dtype>() {
		this->StopInternalThread();
		for (vector<ImageRecord*>::iterator it = lines_.begin(); it != lines_.end(); ++it) {
			if (NULL != *it) {
				delete *it;
				*it = NULL;
			}
		}
		lines_.clear();
	}

	template <typename Dtype>
	void MHDSliceDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		output_labels_ = (top.size() > 1);

		itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());
		typedef itk::ImageFileReader<ImageType> ImageReaderType;
		typedef itk::ImageFileReader<LabelType> LabelReaderType;
		typedef itk::ImageFileWriter<ImageSliceType> ImageSliceWriterType;

		const MHDDataParameter& mhd_data_param = this->layer_param_.mhd_data_param();
		const int batch_size = mhd_data_param.batch_size();
		const string& root_folder = mhd_data_param.root_folder();
		const ContourNameList& contour_name_list = mhd_data_param.contour_name_list();
		contour_num_ = contour_name_list.name_size();
		const string& source = mhd_data_param.source();
		const bool buffer_file_exists = mhd_data_param.buffer_exist();
		LOG(INFO) << "Opening file " << source;

		std::ifstream infile(source.c_str());
		string line;
		size_t pos1, pos2;
		int file_num = 0;
		int epoch_num = 0;
		while (std::getline(infile, line)) {
			LOG(INFO) << "Loading file " << file_num++ << "...";
			pos1 = line.find_first_of(' ');
			pos2 = line.find_last_of(' ');
			string image_file_name = line.substr(0, pos1);
			string label_file_name = line.substr(pos1 + 1, pos2 - pos1 - 1);
			string info_file_name = line.substr(pos2 + 1);
			std::ifstream infile_info(root_folder + info_file_name);
			std::vector<int> contour_labels;
			std::vector<int> roi_x0, roi_x1, roi_y0, roi_y1, roi_z0, roi_z1;
			std::vector<int> exist_contours;
			int label_range[3][2];
			label_range[0][0] = INT_MAX;
			label_range[0][1] = INT_MIN;
			label_range[1][0] = INT_MAX;
			label_range[1][1] = INT_MIN;
			label_range[2][0] = INT_MAX;
			label_range[2][1] = INT_MIN;
			while (std::getline(infile_info, line)) {
				pos1 = string::npos;
				for (int i = 0; i < contour_num_; ++i) {
					pos1 = line.find(contour_name_list.name(i));
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

				pos1 = pos2;
				pos2 = line.find_first_of(' ', pos1 + 1);
				int x0 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
				roi_x0.push_back(x0);
				label_range[0][0] = min(label_range[0][0], x0);

				pos1 = pos2;
				pos2 = line.find_first_of(' ', pos1 + 1);
				int x1 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
				roi_x1.push_back(x1);
				label_range[0][1] = max(label_range[0][1], x1);

				pos1 = pos2;
				pos2 = line.find_first_of(' ', pos1 + 1);
				int y0 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
				roi_y0.push_back(y0);
				label_range[1][0] = min(label_range[1][0], y0);

				pos1 = pos2;
				pos2 = line.find_first_of(' ', pos1 + 1);
				int y1 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
				roi_y1.push_back(y1);
				label_range[1][1] = max(label_range[1][1], y1);

				pos1 = pos2;
				pos2 = line.find_first_of(' ', pos1 + 1);
				int z0 = atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
				roi_z0.push_back(z0);
				label_range[2][0] = min(label_range[2][0], z0);

				pos1 = pos2;
				int z1 = atoi(line.substr(pos1 + 1).c_str());
				roi_z1.push_back(z1);
				label_range[2][1] = max(label_range[2][1], z1);
			}
			if (!contour_labels.empty()) {
				ImageType::DirectionType direct_src;
				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						direct_src[i][j] = (i == j) ? 1 : 0;
					}
				}

				ImageReaderType::Pointer reader_image = ImageReaderType::New();
				reader_image->SetFileName(root_folder + image_file_name);
				if (!buffer_file_exists)
				{
					reader_image->Update();
				}
				else
				{
					reader_image->UpdateOutputInformation();
				}
				ImageType::Pointer image = reader_image->GetOutput();
				image->SetDirection(direct_src);

				if (!buffer_file_exists)
				{
					// rescale intensity to [0, 1]
					const int image_buffer_length = image->GetLargestPossibleRegion().GetNumberOfPixels();
					Dtype* image_buffer = image->GetBufferPointer();
					const Dtype min_intensity = mhd_data_param.min_intensity();
					const Dtype max_intensity = mhd_data_param.max_intensity();
					for (int i = 0; i < image_buffer_length; ++i) {
						if (image_buffer[i] < min_intensity)
							image_buffer[i] = min_intensity;
						if (image_buffer[i] > max_intensity)
							image_buffer[i] = max_intensity;
					}
					double buff_scale = (max_intensity > min_intensity) ? 1.0 / (max_intensity - min_intensity) : 0.0;
					for (int i = 0; i < image_buffer_length; ++i) {
						image_buffer[i] = (image_buffer[i] - min_intensity) * buff_scale;
					}
				}

				//// rescale image stack to [0.0, 1.0]
				//Dtype buff_max = image_buffer[0];
				//Dtype buff_min = image_buffer[0];
				//for (int i = 0; i < image_pixel_num; ++i) {
				//	if (buff_max < image_buffer[i])
				//		buff_max = image_buffer[i];
				//	if (buff_min > image_buffer[i])
				//		buff_min = image_buffer[i];
				//}
				//double buff_scale = (buff_max > buff_min) ? 1.0 / (buff_max - buff_min) : 0.0;
				//for (int i = 0; i < image_pixel_num; ++i) {
				//	image_buffer[i] = (image_buffer[i] - buff_min) * buff_scale;
				//}

				//LabelType::Pointer label;
				//LabelReaderType::Pointer reader_label = LabelReaderType::New();
				//reader_label->SetFileName(root_folder + label_file_name);
				//reader_label->Update();
				//label = reader_label->GetOutput();
				//label->SetDirection(direct_src);

				//int buffer_length = label->GetBufferedRegion().GetNumberOfPixels();
				//char* label_buffer = label->GetBufferPointer();
				//char* label_buffer_tmp = new char[buffer_length];
				//memset(label_buffer_tmp, 0, sizeof(char) * buffer_length);
				//for (int i = 0; i < buffer_length; ++i) {
				//	for (int j = 0; j < contour_labels.size(); ++j) {
				//		if (label_buffer[i] == contour_labels[j]) {
				//			label_buffer_tmp[i] = exist_contours[j];
				//			break;
				//		}
				//	}
				//}
				//memcpy(label_buffer, label_buffer_tmp, sizeof(char) * buffer_length);
				//delete[]label_buffer_tmp;

				const ImageType::SizeType& image_size = image->GetLargestPossibleRegion().GetSize();
				const ImageType::SpacingType& image_spacing = image->GetSpacing();
				const ImageType::PointType& image_origin = image->GetOrigin();

				typedef itk::IdentityTransform<double, 3> IdentityTransformType;
				typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
				typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
				typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;

				// axial
				{
					ImageType::SizeType resample_size = image_size;
					ImageType::SpacingType resample_spacing = image_spacing;
					ImageType::PointType resample_origin = image_origin;

					resample_spacing[0] = mhd_data_param.resample_slice_spacing_x();
					resample_spacing[1] = mhd_data_param.resample_slice_spacing_y();
					resample_size[0] = image_size[0] * image_spacing[0] / resample_spacing[0];
					resample_size[1] = image_size[1] * image_spacing[1] / resample_spacing[1];

					ImageRecord* image_record = new ImageRecord;
					image_record->file_name_ = image_file_name;
					image_record->direction_ = 0;
					image_record->slice_size_[0] = resample_size[0];
					image_record->slice_size_[1] = resample_size[1];
					image_record->slice_spacing_[0] = resample_spacing[0];
					image_record->slice_spacing_[1] = resample_spacing[1];
					image_record->length_ = resample_size[2];
					image_record->seek_ = 0;
					lines_.push_back(image_record);

					epoch_num += image_record->length_ / batch_size + 1;

					if (!buffer_file_exists)
					{
						ResampleImageFilterType::Pointer resampler = ResampleImageFilterType::New();
						resampler->SetInput(image);
						resampler->SetSize(resample_size);
						resampler->SetOutputSpacing(resample_spacing);
						resampler->SetOutputOrigin(resample_origin);
						resampler->SetTransform(IdentityTransformType::New());
						resampler->Update();
						ImageType::Pointer resample_volume = resampler->GetOutput();
						Dtype* resample_volume_buffer = resample_volume->GetBufferPointer();

						for (int i = 0; i < resample_size[2]; ++i)
						{
							stringstream slice_file_name;
							slice_file_name << mhd_data_param.buffer_path() << image_file_name << ".A." << i << ".mhd";

							ImageSliceType::RegionType slice_region;
							ImageSliceType::SizeType slice_size;
							slice_size[0] = resample_size[0];
							slice_size[1] = resample_size[1];
							slice_region.SetSize(slice_size);
							ImageSliceType::SpacingType slice_spacing;
							slice_spacing[0] = resample_spacing[0];
							slice_spacing[1] = resample_spacing[1];
							ImageSliceType::Pointer slice = ImageSliceType::New();
							slice->SetRegions(slice_region);
							slice->SetSpacing(slice_spacing);
							slice->Allocate();
							Dtype* slice_buffer = slice->GetBufferPointer();

							for (int j = 0; j < slice_size[1]; ++j)
							{
								for (int k = 0; k < slice_size[0]; ++k)
								{
									slice_buffer[j * slice_size[0] + k] = resample_volume_buffer[i * resample_size[1] * resample_size[0] + j * resample_size[0] + k];
								}
							}

							ImageSliceWriterType::Pointer writer = ImageSliceWriterType::New();
							writer->SetFileName(slice_file_name.str());
							writer->SetInput(slice);
							writer->Update();

							std::vector<int> label_exist;
							label_exist.resize(contour_num_, 0);
							for (int j = 0; j < exist_contours.size(); ++j)
							{
								if (i >= roi_z0[j] && i <= roi_z1[j])
								{
									label_exist[exist_contours[j] - 1] = 1;
								}
							}

							SliceRecord* slice_record = new SliceRecord;
							slice_record->file_name_ = slice_file_name.str();
							slice_record->label_exist_.swap(label_exist);
							image_record->slice_.push_back(slice_record);
						}
					}
					else
					{
						for (int i = 0; i < resample_size[2]; ++i)
						{
							stringstream slice_file_name;
							slice_file_name << mhd_data_param.buffer_path() << image_file_name << ".A." << i << ".mhd";

							std::vector<int> label_exist;
							label_exist.resize(contour_num_, 0);
							for (int j = 0; j < exist_contours.size(); ++j)
							{
								if (i >= roi_z0[j] && i <= roi_z1[j])
								{
									label_exist[exist_contours[j] - 1] = 1;
								}
							}

							SliceRecord* slice_record = new SliceRecord;
							slice_record->file_name_ = slice_file_name.str();
							slice_record->label_exist_.swap(label_exist);
							image_record->slice_.push_back(slice_record);
						}
					}

					if (this->phase_ == TRAIN && mhd_data_param.shuffle()) {
						const unsigned int prefetch_rng_seed = caffe_rng_rand();
						prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
						caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
						shuffle(image_record->slice_.begin(), image_record->slice_.end(), prefetch_rng);
					}
				}


				// coronal
				{
					ImageType::SizeType resample_size = image_size;
					ImageType::SpacingType resample_spacing = image_spacing;
					ImageType::PointType resample_origin = image_origin;

					resample_spacing[0] = mhd_data_param.resample_slice_spacing_x();
					resample_spacing[2] = mhd_data_param.resample_slice_spacing_y();
					resample_size[0] = image_size[0] * image_spacing[0] / resample_spacing[0];
					resample_size[2] = image_size[2] * image_spacing[2] / resample_spacing[2];

					ImageRecord* image_record = new ImageRecord;
					image_record->file_name_ = image_file_name;
					image_record->direction_ = 1;
					image_record->slice_size_[0] = resample_size[0];
					image_record->slice_size_[1] = resample_size[2];
					image_record->slice_spacing_[0] = resample_spacing[0];
					image_record->slice_spacing_[1] = resample_spacing[2];
					image_record->length_ = resample_size[1];
					image_record->seek_ = 0;
					lines_.push_back(image_record);

					epoch_num += image_record->length_ / batch_size + 1;

					if (!buffer_file_exists)
					{
						ResampleImageFilterType::Pointer resampler = ResampleImageFilterType::New();
						resampler->SetInput(image);
						resampler->SetSize(resample_size);
						resampler->SetOutputSpacing(resample_spacing);
						resampler->SetOutputOrigin(resample_origin);
						resampler->SetTransform(IdentityTransformType::New());
						resampler->Update();
						ImageType::Pointer resample_volume = resampler->GetOutput();
						Dtype* resample_volume_buffer = resample_volume->GetBufferPointer();

						for (int i = 0; i < resample_size[1]; ++i)
						{
							stringstream slice_file_name;
							slice_file_name << mhd_data_param.buffer_path() << image_file_name << ".C." << i << ".mhd";

							ImageSliceType::Pointer slice = ImageSliceType::New();
							ImageSliceType::RegionType slice_region;
							ImageSliceType::SizeType slice_size;
							slice_size[0] = resample_size[0];
							slice_size[1] = resample_size[2];
							slice_region.SetSize(slice_size);
							ImageSliceType::SpacingType slice_spacing;
							slice_spacing[0] = resample_spacing[0];
							slice_spacing[1] = resample_spacing[2];
							slice->SetRegions(slice_region);
							slice->SetSpacing(slice_spacing);
							slice->Allocate();
							Dtype* slice_buffer = slice->GetBufferPointer();

							for (int j = 0; j < slice_size[1]; ++j)
							{
								for (int k = 0; k < slice_size[0]; ++k)
								{
									slice_buffer[j * slice_size[0] + k] = resample_volume_buffer[j * resample_size[1] * resample_size[0] + i * resample_size[0] + k];
								}
							}

							ImageSliceWriterType::Pointer writer = ImageSliceWriterType::New();
							writer->SetFileName(slice_file_name.str());
							writer->SetInput(slice);
							writer->Update();

							std::vector<int> label_exist;
							label_exist.resize(contour_num_, 0);
							for (int j = 0; j < exist_contours.size(); ++j)
							{
								if (i >= roi_y0[j] && i <= roi_y1[j])
								{
									label_exist[exist_contours[j] - 1] = 1;
								}
							}

							SliceRecord* slice_record = new SliceRecord;
							slice_record->file_name_ = slice_file_name.str();
							slice_record->label_exist_.swap(label_exist);
							image_record->slice_.push_back(slice_record);
						}
					}
					else
					{
						for (int i = 0; i < resample_size[1]; ++i)
						{
							stringstream slice_file_name;
							slice_file_name << mhd_data_param.buffer_path() << image_file_name << ".C." << i << ".mhd";

							std::vector<int> label_exist;
							label_exist.resize(contour_num_, 0);
							for (int j = 0; j < exist_contours.size(); ++j)
							{
								if (i >= roi_y0[j] && i <= roi_y1[j])
								{
									label_exist[exist_contours[j] - 1] = 1;
								}
							}

							SliceRecord* slice_record = new SliceRecord;
							slice_record->file_name_ = slice_file_name.str();
							slice_record->label_exist_.swap(label_exist);
							image_record->slice_.push_back(slice_record);
						}
					}

					if (this->phase_ == TRAIN && mhd_data_param.shuffle()) {
						const unsigned int prefetch_rng_seed = caffe_rng_rand();
						prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
						caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
						shuffle(image_record->slice_.begin(), image_record->slice_.end(), prefetch_rng);
					}
				}

				// sagittal
				{
					ImageType::SizeType resample_size = image_size;
					ImageType::SpacingType resample_spacing = image_spacing;
					ImageType::PointType resample_origin = image_origin;

					resample_spacing[1] = mhd_data_param.resample_slice_spacing_x();
					resample_spacing[2] = mhd_data_param.resample_slice_spacing_y();
					resample_size[1] = image_size[1] * image_spacing[1] / resample_spacing[1];
					resample_size[2] = image_size[2] * image_spacing[2] / resample_spacing[2];

					ImageRecord* image_record = new ImageRecord;
					image_record->file_name_ = image_file_name;
					image_record->direction_ = 2;
					image_record->slice_size_[0] = resample_size[1];
					image_record->slice_size_[1] = resample_size[2];
					image_record->slice_spacing_[0] = resample_spacing[1];
					image_record->slice_spacing_[1] = resample_spacing[2];
					image_record->length_ = resample_size[0];
					image_record->seek_ = 0;
					lines_.push_back(image_record);

					epoch_num += image_record->length_ / batch_size + 1;

					if (!buffer_file_exists)
					{
						ResampleImageFilterType::Pointer resampler = ResampleImageFilterType::New();
						resampler->SetInput(image);
						resampler->SetSize(resample_size);
						resampler->SetOutputSpacing(resample_spacing);
						resampler->SetOutputOrigin(resample_origin);
						resampler->SetTransform(IdentityTransformType::New());
						resampler->Update();
						ImageType::Pointer resample_volume = resampler->GetOutput();
						Dtype* resample_volume_buffer = resample_volume->GetBufferPointer();

						for (int i = 0; i < resample_size[0]; ++i)
						{
							stringstream slice_file_name;
							slice_file_name << mhd_data_param.buffer_path() << image_file_name << ".S." << i << ".mhd";

							ImageSliceType::Pointer slice = ImageSliceType::New();
							ImageSliceType::RegionType slice_region;
							ImageSliceType::SizeType slice_size;
							slice_size[0] = resample_size[1];
							slice_size[1] = resample_size[2];
							slice_region.SetSize(slice_size);
							ImageSliceType::SpacingType slice_spacing;
							slice_spacing[0] = resample_spacing[1];
							slice_spacing[1] = resample_spacing[2];
							slice->SetRegions(slice_region);
							slice->SetSpacing(slice_spacing);
							slice->Allocate();
							Dtype* slice_buffer = slice->GetBufferPointer();

							for (int j = 0; j < slice_size[1]; ++j)
							{
								for (int k = 0; k < slice_size[0]; ++k)
								{
									slice_buffer[j * slice_size[0] + k] = resample_volume_buffer[j * resample_size[1] * resample_size[0] + k * resample_size[0] + i];
								}
							}

							ImageSliceWriterType::Pointer writer = ImageSliceWriterType::New();
							writer->SetFileName(slice_file_name.str());
							writer->SetInput(slice);
							writer->Update();

							std::vector<int> label_exist;
							label_exist.resize(contour_num_, 0);
							for (int j = 0; j < exist_contours.size(); ++j)
							{
								if (i >= roi_x0[j] && i <= roi_x1[j])
								{
									label_exist[exist_contours[j] - 1] = 1;
								}
							}

							SliceRecord* slice_record = new SliceRecord;
							slice_record->file_name_ = slice_file_name.str();
							slice_record->label_exist_.swap(label_exist);
							image_record->slice_.push_back(slice_record);
						}
					}
					else
					{
						for (int i = 0; i < resample_size[0]; ++i)
						{
							stringstream slice_file_name;
							slice_file_name << mhd_data_param.buffer_path() << image_file_name << ".S." << i << ".mhd";

							std::vector<int> label_exist;
							label_exist.resize(contour_num_, 0);
							for (int j = 0; j < exist_contours.size(); ++j)
							{
								if (i >= roi_x0[j] && i <= roi_x1[j])
								{
									label_exist[exist_contours[j] - 1] = 1;
								}
							}

							SliceRecord* slice_record = new SliceRecord;
							slice_record->file_name_ = slice_file_name.str();
							slice_record->label_exist_.swap(label_exist);
							image_record->slice_.push_back(slice_record);
						}
					}

					if (this->phase_ == TRAIN && mhd_data_param.shuffle()) {
						const unsigned int prefetch_rng_seed = caffe_rng_rand();
						prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
						caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
						shuffle(image_record->slice_.begin(), image_record->slice_.end(), prefetch_rng);
					}
				}
			}
		}

		CHECK(!lines_.empty()) << "File is empty";

		if (this->phase_ == TRAIN && mhd_data_param.shuffle()) {
			// randomly shuffle data
			LOG(INFO) << "Shuffling data";
			const unsigned int prefetch_rng_seed = caffe_rng_rand();
			prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
			ShuffleImages();
		}
		else {
			if (this->phase_ == TRAIN && mhd_data_param.rand_skip() == 0) {
				LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
			}
		}
		LOG(INFO) << "A total of " << lines_.size() << " images, " << epoch_num << " iterations per epoch.";

		lines_id_ = 0;
		// Check if we would need to randomly skip a few data points
		if (mhd_data_param.rand_skip()) {
			unsigned int skip = caffe_rng_rand() % mhd_data_param.rand_skip();
			LOG(INFO) << "Skipping first " << skip << " data points.";
			CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
			lines_id_ = skip;
		}

		const unsigned int ind_rng_seed = caffe_rng_rand();
		ind_rng_.reset(new Caffe::RNG(ind_rng_seed));
		const unsigned int trans_rng_seed = caffe_rng_rand();
		trans_rng_.reset(new Caffe::RNG(trans_rng_seed));

		vector<int> data_shape(4);
		data_shape[0] = 1;
		data_shape[1] = 1;
		data_shape[2] = 512;
		data_shape[3] = 512;
		vector<int> label_shape(2);
		label_shape[0] = 1;
		label_shape[1] = contour_num_;
		for (int i = 0; i < PREFETCH_COUNT; ++i) {
			this->prefetch_[i].data_.Reshape(data_shape);
			if (output_labels_) this->prefetch_[i].label_.Reshape(label_shape);
		}

		top[0]->Reshape(data_shape);
		if (output_labels_) top[1]->Reshape(label_shape);
	}

	template <typename Dtype>
	void MHDSliceDataLayer<Dtype>::ShuffleImages() {
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		shuffle(lines_.begin(), lines_.end(), prefetch_rng);
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void MHDSliceDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		typedef itk::ImageFileReader<ImageType> ImageReaderType;
		typedef itk::ImageFileReader<LabelType> LabelReaderType;
		typedef itk::ImageFileWriter<ImageType> ImageWriterType;
		typedef itk::ImageFileWriter<LabelType> LabelWriterType;
		typedef itk::ImageFileReader<ImageSliceType> ImageSliceReaderType;
		typedef itk::ImageFileReader<LabelSliceType> LabelSliceReaderType;
		typedef itk::ImageFileWriter<ImageSliceType> ImageSliceWriterType;
		typedef itk::ImageFileWriter<LabelSliceType> LabelSliceWriterType;
		typedef itk::HistogramMatchingImageFilter<ImageType, ImageType> HMFilterType;
		typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
		typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
		typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;
		typedef itk::ResampleImageFilter<ImageSliceType, ImageSliceType> ResampleSliceFilterType;
		typedef itk::IdentityTransform<double, 2> IdentityTransformType;
		typedef itk::AffineTransform<double, 2> RotateTransformType;

		const int lines_size = lines_.size();
		CHECK_GT(lines_size, lines_id_);
		ImageRecord* image_record = lines_[lines_id_];

		const MHDDataParameter& mhd_data_param = this->layer_param_.mhd_data_param();
		const int batch_size = std::min((int)mhd_data_param.batch_size(), image_record->length_ - image_record->seek_);
		const string& root_folder = mhd_data_param.root_folder();
		int max_slice_size[2] = { 0, 0 };
		max_slice_size[0] = image_record->slice_size_[0];
		max_slice_size[1] = image_record->slice_size_[1];

		vector<int> data_shape(4);
		data_shape[0] = batch_size;
		data_shape[1] = 1;
		data_shape[2] = max_slice_size[1];
		data_shape[3] = max_slice_size[0];
		batch->data_.Reshape(data_shape);
		Dtype* batch_data = batch->data_.mutable_cpu_data();
		memset(batch_data, 0, sizeof(Dtype) * batch_size * max_slice_size[1] * max_slice_size[0]);

		Dtype* label_data = NULL;
		if (output_labels_)
		{
			vector<int> label_shape(3);
			label_shape[0] = batch_size;
			label_shape[1] = 1;
			label_shape[2] = contour_num_;
			batch->label_.Reshape(label_shape);
			label_data = batch->label_.mutable_cpu_data();
		}

		for (int batch_id = 0; batch_id < batch_size; ++batch_id)
		{
			SliceRecord* slice_record = image_record->slice_[image_record->seek_++];
			ImageSliceReaderType::Pointer reader = ImageSliceReaderType::New();
			reader->SetFileName(slice_record->file_name_);
			reader->Update();
			ImageSliceType::Pointer slice = reader->GetOutput();

			const ImageSliceType::SizeType& slice_size = slice->GetBufferedRegion().GetSize();
			const ImageSliceType::SpacingType& slice_spacing = slice->GetSpacing();
			const ImageSliceType::PointType& slice_origin = slice->GetOrigin();

			ImageSliceType::SizeType resample_size = slice_size;
			ImageSliceType::SpacingType resample_spacing = slice_spacing;
			ImageSliceType::PointType resample_origin = slice_origin;

			resample_size[0] = max_slice_size[0];
			resample_size[1] = max_slice_size[1];
			resample_origin[0] = slice_origin[0] + 0.5 * slice_size[0] * slice_spacing[0] - 0.5 * resample_size[0] * resample_spacing[0];
			resample_origin[1] = slice_origin[1] + 0.5 * slice_size[1] * slice_spacing[1] - 0.5 * resample_size[1] * resample_spacing[1];

			ResampleSliceFilterType::Pointer resampler = ResampleSliceFilterType::New();
			resampler->SetInput(slice);
			resampler->SetSize(resample_size);
			resampler->SetOutputSpacing(resample_spacing);
			resampler->SetOutputOrigin(resample_origin);

			RotateTransformType::Pointer rotate_transform = RotateTransformType::New();
			RotateTransformType::OutputVectorType translation;
			translation[0] = -(image_record->slice_spacing_[0] * image_record->slice_size_[0] * 0.5);
			translation[1] = -(image_record->slice_spacing_[1] * image_record->slice_size_[1] * 0.5);
			rotate_transform->Translate(translation);
			const unsigned int r = caffe_rng_rand();
			int angle_in_degree = r % 21 - 10;
			double angle = angle_in_degree * 3.141592654 / 180.0;
			rotate_transform->Rotate2D(angle, false);
			translation[0] = -translation[0];
			translation[1] = -translation[1];
			rotate_transform->Translate(translation, false);

			if (this->phase_ == TRAIN)
			{
				resampler->SetTransform(rotate_transform);
			}
			else
			{
				resampler->SetTransform(IdentityTransformType::New());
			}
			resampler->Update();
			ImageSliceType::Pointer resample_slice = resampler->GetOutput();

			//ImageSliceWriterType::Pointer slice_writer = ImageSliceWriterType::New();
			//slice_writer->SetFileName("F:/slice.mhd");
			//slice_writer->SetInput(resample_slice);
			//slice_writer->Update();

			Dtype* slice_buffer = resample_slice->GetBufferPointer();
			memcpy(batch_data + batch_id * max_slice_size[1] * max_slice_size[0], slice_buffer, sizeof(Dtype) * max_slice_size[1] * max_slice_size[0]);
			if (output_labels_)
			{
				for (int i = 0; i < contour_num_; ++i)
				{
					label_data[batch_id * contour_num_ + i] = slice_record->label_exist_[i];
				}
			}
		}
		if (this->phase_ == TRAIN)
		{
			if (image_record->seek_ >= image_record->length_)
			{
				// reset slice seek
				image_record->seek_ = 0;

				const unsigned int prefetch_rng_seed = caffe_rng_rand();
				prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
				caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
				shuffle(image_record->slice_.begin(), image_record->slice_.end(), prefetch_rng);
			}

			// go to the next iter
			lines_id_++;
			if (lines_id_ >= lines_size) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				lines_id_ = 0;
				if (this->layer_param_.mhd_data_param().shuffle()) {
					ShuffleImages();
				}
			}
		}
		else
		{
			if (image_record->seek_ >= image_record->length_)
			{
				// reset slice seek
				image_record->seek_ = 0;

				// go to the next iter
				lines_id_++;
				if (lines_id_ >= lines_size) {
					// We have reached the end. Restart from the first.
					DLOG(INFO) << "Restarting data prefetching from start.";
					lines_id_ = 0;
				}
			}
		}
	}

	INSTANTIATE_CLASS(MHDSliceDataLayer);
	REGISTER_LAYER_CLASS(MHDSliceData);

}  // namespace caffe
