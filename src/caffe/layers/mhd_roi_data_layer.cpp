#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/mhd_roi_data_layer.hpp"
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

using std::floor;

namespace caffe {

template <typename Dtype>
MHDRoiDataLayer<Dtype>::~MHDRoiDataLayer<Dtype>() {
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
void MHDRoiDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());
	typedef itk::ImageFileReader<ImageType> ImageReaderType;
	typedef itk::ImageFileReader<LabelType> LabelReaderType;

	const MHDDataParameter& mhd_data_param = this->layer_param_.mhd_data_param();
	const string& root_folder = mhd_data_param.root_folder();
	const ContourNameList& contour_name_list = mhd_data_param.contour_name_list();
	const string& source = mhd_data_param.source();
	LOG(INFO) << "Opening file " << source;

	std::ifstream infile(source.c_str());
	string line;
	size_t pos1, pos2;
	int file_num = 0;
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
			for (int i = 0; i < contour_name_list.name_size(); ++i) {
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

			ImageRecord* image_record = new ImageRecord;
			image_record->roi_label_.swap(exist_contours);
			image_record->roi_x0_.swap(roi_x0);
			image_record->roi_x1_.swap(roi_x1);
			image_record->roi_y0_.swap(roi_y0);
			image_record->roi_y1_.swap(roi_y1);
			image_record->roi_z0_.swap(roi_z0);
			image_record->roi_z1_.swap(roi_z1);

			ImageReaderType::Pointer reader_image = ImageReaderType::New();
			reader_image->SetFileName(root_folder + image_file_name);
			reader_image->Update();
			ImageType::Pointer image = reader_image->GetOutput();
			image->SetDirection(direct_src);

			LabelType::Pointer label;
			if (output_labels_) {
				LabelReaderType::Pointer reader_label = LabelReaderType::New();
				reader_label->SetFileName(root_folder + label_file_name);
				reader_label->Update();
				label = reader_label->GetOutput();
				label->SetDirection(direct_src);

				int buffer_length = label->GetBufferedRegion().GetNumberOfPixels();
				char* label_buffer = label->GetBufferPointer();
				char* label_buffer_tmp = new char[buffer_length];
				memset(label_buffer_tmp, 0, sizeof(char) * buffer_length);
				for (int i = 0; i < buffer_length; ++i) {
					for (int j = 0; j < contour_labels.size(); ++j) {
						if (label_buffer[i] >= contour_labels[j]) {
							label_buffer_tmp[i] = exist_contours[j];
							break;
						}
					}
				}
				memcpy(label_buffer, label_buffer_tmp, sizeof(char) * buffer_length);
				delete[]label_buffer_tmp;
			}

			// resample
			{
				const ImageType::SizeType& origin_size = image->GetBufferedRegion().GetSize();
				const ImageType::SpacingType& origin_spacing = image->GetSpacing();
				const ImageType::PointType& origin_origin = image->GetOrigin();

				ImageType::SizeType resample_size;
				ImageType::SpacingType resample_spacing;
				ImageType::PointType resample_origin;

				resample_spacing[0] = mhd_data_param.resample_spacing_x();
				resample_spacing[1] = mhd_data_param.resample_spacing_y();
				resample_spacing[2] = mhd_data_param.resample_spacing_z();
				resample_size[0] = origin_size[0] * origin_spacing[0] / resample_spacing[0];
				resample_size[1] = origin_size[1] * origin_spacing[1] / resample_spacing[1];
				resample_size[2] = origin_size[2] * origin_spacing[2] / resample_spacing[2];
				resample_size[0] = std::min((int)resample_size[0], (int)mhd_data_param.max_width());
				resample_size[1] = std::min((int)resample_size[1], (int)mhd_data_param.max_height());
				resample_size[2] = std::min((int)resample_size[2], (int)mhd_data_param.max_length());
				resample_origin[0] = origin_origin[0] + origin_size[0] * origin_spacing[0] * 0.5 - resample_size[0] * resample_spacing[0] * 0.5;
				resample_origin[1] = origin_origin[1] + origin_size[1] * origin_spacing[1] * 0.5 - resample_size[1] * resample_spacing[1] * 0.5;
				resample_origin[2] = origin_origin[2] + origin_size[2] * origin_spacing[2] * 0.5 - resample_size[2] * resample_spacing[2] * 0.5;

				for (int i = 0; i < 3; ++i)	{
					image_record->origin_size_[i] = origin_size[i];
					image_record->origin_spacing_[i] = origin_spacing[i];
					image_record->origin_origin_[i] = origin_origin[i];
					image_record->size_[i] = resample_size[i];
					image_record->spacing_[i] = resample_spacing[i];
					image_record->origin_[i] = resample_origin[i];
					for (int j = 0; j < 2; ++j) {
						image_record->label_range_[i][j] = label_range[i][j];
					}
				}

				typedef itk::IdentityTransform<double, 3> IdentityTransformType;
				typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
				typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
				typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;

				ResampleImageFilterType::Pointer resampler_image = ResampleImageFilterType::New();
				resampler_image->SetInput(image);
				resampler_image->SetSize(resample_size);
				resampler_image->SetOutputSpacing(resample_spacing);
				resampler_image->SetOutputOrigin(resample_origin);
				resampler_image->SetTransform(IdentityTransformType::New());
				resampler_image->Update();
				image = resampler_image->GetOutput();

				if (output_labels_) {
					ResampleLabelFilterType::Pointer resampler_label = ResampleLabelFilterType::New();
					resampler_label->SetInput(label);
					resampler_label->SetSize(resample_size);
					resampler_label->SetOutputSpacing(resample_spacing);
					resampler_label->SetOutputOrigin(resample_origin);
					resampler_label->SetTransform(IdentityTransformType::New());
					resampler_label->SetInterpolator(InterpolatorType::New());
					resampler_label->Update();
					label = resampler_label->GetOutput();
				}
			}

			// rescale intensity to [0, 1]
			const int image_buffer_length = image->GetBufferedRegion().GetNumberOfPixels();
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

			image_record->image_ = image;
			image_record->label_ = label;
			image_record->info_file_name_ = info_file_name;
			lines_.push_back(image_record);
		}
	}

	CHECK(!lines_.empty()) << "File is empty";

	if (this->phase_ == TRAIN && mhd_data_param.shuffle()) {
		// randomly shuffle data
		LOG(INFO) << "Shuffling data";
		const unsigned int prefetch_rng_seed = caffe_rng_rand();
		prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
		ShuffleImages();
	} else {
		if (this->phase_ == TRAIN &&
			mhd_data_param.rand_skip() == 0) {
			LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
		}
	}
	LOG(INFO) << "A total of " << lines_.size() << " images.";

	lines_id_ = 0;
	iter_id_ = 0;
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

	vector<int> data_shape(5);
	data_shape[0] = 1;
	data_shape[1] = 1;
	data_shape[2] = 64;
	data_shape[3] = 128;
	data_shape[4] = 128;
	vector<int> info_shape(2);
	info_shape[0] = 1;
	info_shape[1] = 19;
	vector<int> roi_shape(2);
	roi_shape[0] = 1;
	roi_shape[1] = 8;
	for (int i = 0; i < PREFETCH_COUNT; ++i) {
		this->prefetch_[i].data_.Reshape(data_shape);
		this->prefetch_[i].info_.Reshape(info_shape);
		if (output_roi_) this->prefetch_[i].roi_.Reshape(roi_shape);
		if (output_labels_) this->prefetch_[i].label_.Reshape(data_shape);
	}

	top[0]->Reshape(data_shape);
	top[1]->Reshape(info_shape);
	if (output_roi_) top[2]->Reshape(roi_shape);
	if (output_labels_) top[3]->Reshape(data_shape);
}

template <typename Dtype>
void MHDRoiDataLayer<Dtype>::ShuffleImages() {
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void MHDRoiDataLayer<Dtype>::load_batch(RoiBatch<Dtype>* batch) {

	typedef itk::ImageFileReader<ImageType> ImageReaderType;
	typedef itk::ImageFileReader<LabelType> LabelReaderType;
	typedef itk::ImageFileWriter<ImageType> ImageWriterType;
	typedef itk::ImageFileWriter<LabelType> LabelWriterType;
	typedef itk::HistogramMatchingImageFilter<ImageType, ImageType> HMFilterType;
	typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
	typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
	typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;

	const MHDDataParameter& mhd_data_param = this->layer_param_.mhd_data_param();
	const int lines_size = lines_.size();
	CHECK_GT(lines_size, lines_id_);
	const ImageRecord* image_record = lines_[lines_id_];
	//LOG(INFO) << "Iteration " << iter_id_ << ": " << image_record->info_file_name_;
	ImageType::Pointer src_image = image_record->image_;
	const ImageType::RegionType& src_image_region = src_image->GetBufferedRegion();
	const ImageType::SizeType& src_image_size = src_image->GetBufferedRegion().GetSize();
	const ImageType::SpacingType& src_image_spacing = src_image->GetSpacing();
	const ImageType::PointType& src_image_origin = src_image->GetOrigin();
	const ImageType::DirectionType& src_image_direct = src_image->GetDirection();

	const int batch_size = mhd_data_param.batch_size();
	const int shift_offset[3][7] = {
		0, -5, 5, 0, 0, 0, 0,
		0, 0, 0, -5, 5, 0, 0,
		0, 0, 0, 0, 0, -1, 1
	};
	const int roi_num = image_record->roi_label_.size();

	const int min_image_length = mhd_data_param.min_truncate_length();
	const double crop_prob = (caffe_rng_rand() % 100) / 99.0;
	const bool need_crop = (this->phase_ == TRAIN && crop_prob < mhd_data_param.truncate_probability() && src_image_size[2] > min_image_length);
	const unsigned int crop_length = need_crop ? caffe_rng_rand() % (src_image_size[2] - min_image_length) + min_image_length : src_image_size[2];

	vector<int> data_shape(5);
	data_shape[0] = batch_size;
	data_shape[1] = 1;
	data_shape[2] = crop_length;
	data_shape[3] = src_image_size[1];
	data_shape[4] = src_image_size[0];
	batch->data_.Reshape(data_shape);

	std::vector<int> data_info_shape(2);
	data_info_shape[0] = batch_size;
	data_info_shape[1] = 19;
	batch->info_.Reshape(data_info_shape);

	if (output_roi_) {
		std::vector<int> roi_shape(3);
		roi_shape[0] = batch_size;
		roi_shape[1] = roi_num;
		roi_shape[2] = 8;
		batch->roi_.Reshape(roi_shape);
	}

	if (output_labels_) {
		batch->label_.Reshape(data_shape);
	}

	for (int batch_id = 0; batch_id < batch_size; ++batch_id) {

		ImageType::RegionType image_region = src_image_region;
		ImageType::SizeType image_size = src_image_size;
		ImageType::SpacingType image_spacing = src_image_spacing;
		ImageType::PointType image_origin = src_image_origin;
		ImageType::DirectionType image_direct = src_image_direct;

		if (need_crop) {
			const unsigned int crop_start = caffe_rng_rand() % (src_image_size[2] - crop_length + 1);
			image_size[2] = crop_length;
			image_region.SetSize(image_size);
			image_origin[2] = src_image_origin[2] + crop_start * src_image_spacing[2];
		}

		const int inplane_shift = mhd_data_param.inplane_shift();
		if (this->phase_ == TRAIN && inplane_shift > 0) {
			const int shift_x = caffe_rng_rand() % (inplane_shift * 2 + 1) - inplane_shift;
			const int shift_y = caffe_rng_rand() % (inplane_shift * 2 + 1) - inplane_shift;
			image_origin[0] = src_image_origin[0] + shift_x * src_image_spacing[0];
			image_origin[1] = src_image_origin[1] + shift_y * src_image_spacing[1];
		}
		
		if (this->phase_ == TEST) {
			const int shift_x = shift_offset[0][batch_id];
			const int shift_y = shift_offset[1][batch_id];
			const int shift_z = shift_offset[2][batch_id];
			image_origin[0] = src_image_origin[0] + shift_x * src_image_spacing[0];
			image_origin[1] = src_image_origin[1] + shift_y * src_image_spacing[1];
			image_origin[2] = src_image_origin[2] + shift_z * src_image_spacing[2];
		}

		typedef itk::IdentityTransform<double, 3> IdentityTransformType;
		typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
		typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
		typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;

		ResampleImageFilterType::Pointer image_resampler = ResampleImageFilterType::New();
		image_resampler->SetInput(src_image);
		image_resampler->SetSize(image_size);
		image_resampler->SetOutputSpacing(image_spacing);
		image_resampler->SetOutputOrigin(image_origin);
		image_resampler->SetTransform(IdentityTransformType::New());
		image_resampler->Update();
		ImageType::Pointer image = image_resampler->GetOutput();
		Dtype* image_buffer = image->GetBufferPointer();

		char* label_buffer = NULL;
		LabelType::Pointer label;
		if (output_labels_) {
			ResampleLabelFilterType::Pointer label_resampler = ResampleLabelFilterType::New();
			label_resampler->SetInput(image_record->label_);
			label_resampler->SetSize(image_size);
			label_resampler->SetOutputSpacing(image_spacing);
			label_resampler->SetOutputOrigin(image_origin);
			label_resampler->SetTransform(IdentityTransformType::New());
			label_resampler->SetInterpolator(InterpolatorType::New());
			label_resampler->Update();
			label = label_resampler->GetOutput();
		}

		const int pixel_num = image_size[0] * image_size[1] * image_size[2];
		const int slice_pixel_num = image_size[0] * image_size[1];

		// histogram matching
		if (this->phase_ == TRAIN && mhd_data_param.hist_matching()) {
			caffe::rng_t* rng = static_cast<caffe::rng_t*>(ind_rng_->generator());
			const int lines_next_id = (*rng)() % lines_size;
			const ImageRecord* next_image_record = lines_[lines_next_id];
			ImageType::Pointer image_hist = next_image_record->image_;

			HMFilterType::Pointer hist_matching = HMFilterType::New();
			hist_matching->SetReferenceImage(image_hist);
			hist_matching->SetInput(src_image);
			hist_matching->SetNumberOfHistogramLevels(80);
			hist_matching->SetNumberOfMatchPoints(12);
			hist_matching->ThresholdAtMeanIntensityOn();
			hist_matching->Update();
			image = hist_matching->GetOutput();
			image_buffer = image->GetBufferPointer();
		}

		//// rescale intensities to [0.0, 1.0]
		//const Dtype min_intensity = mhd_data_param.min_intensity();
		//const Dtype max_intensity = mhd_data_param.max_intensity();
		//Dtype buff_max = min_intensity;
		//Dtype buff_min = max_intensity;
		//for (int i = 0; i < pixel_num; ++i) {
		//	if (image_buffer[i] < min_intensity)
		//		image_buffer[i] = min_intensity;
		//	if (image_buffer[i] > max_intensity)
		//		image_buffer[i] = max_intensity;
		//	if (buff_max < image_buffer[i])
		//		buff_max = image_buffer[i];
		//	if (buff_min > image_buffer[i])
		//		buff_min = image_buffer[i];
		//}
		//double buff_scale = (buff_max > buff_min) ? 1.0 / (buff_max - buff_min) : 0.0;
		//for (int i = 0; i < pixel_num; ++i) {
		//	image_buffer[i] = (image_buffer[i] - buff_min) * buff_scale;
		//}

		// elastic deformation in slice plane
		caffe::rng_t* rng = static_cast<caffe::rng_t*>(trans_rng_->generator());
		if (this->phase_ == TRAIN && mhd_data_param.random_deform() > 0 &&
			((*rng)() % 1001) / 1000.0 >= 1 - mhd_data_param.random_deform()) {

			typedef itk::Image<Dtype, 2> SliceImageType;
			typedef itk::Image<char, 2> SliceLabelType;
			SliceImageType::Pointer image_stack_slice = SliceImageType::New();
			SliceImageType::SizeType size_slice;
			SliceImageType::RegionType region_slice;
			SliceImageType::PointType origin_slice;
			SliceImageType::SpacingType spacing_slice;
			size_slice[0] = image_size[0];
			size_slice[1] = image_size[1];
			region_slice.SetSize(size_slice);
			spacing_slice[0] = image_spacing[0];
			spacing_slice[1] = image_spacing[1];
			origin_slice[0] = image_origin[0];
			origin_slice[1] = image_origin[1];
			image_stack_slice->SetSpacing(spacing_slice);
			image_stack_slice->SetOrigin(origin_slice);
			image_stack_slice->SetRegions(region_slice);
			image_stack_slice->Allocate();

			typedef itk::BSplineTransform<double, 2, 3> BSplineTransformType;
			typedef itk::BSplineTransformInitializer<BSplineTransformType, SliceImageType> InitializerType;
			typedef itk::NearestNeighborInterpolateImageFunction<SliceLabelType, double> DeformInterpolatorType;
			BSplineTransformType::Pointer transform = BSplineTransformType::New();
			BSplineTransformType::MeshSizeType mesh_size;
			mesh_size.Fill(mhd_data_param.deform_control_point());
			InitializerType::Pointer trans_initial = InitializerType::New();
			trans_initial->SetImage(image_stack_slice);
			trans_initial->SetTransform(transform);
			trans_initial->SetTransformDomainMeshSize(mesh_size);
			trans_initial->InitializeTransform();
			BSplineTransformType::ParametersType trans_param = transform->GetParameters();
			trans_param.GetSize();
			for (int i = 0; i < trans_param.GetNumberOfElements(); ++i) {
				caffe::rng_t* rng = static_cast<caffe::rng_t*>(trans_rng_->generator());
				trans_param.SetElement(i, trans_param.GetElement(i) + (((*rng)() % 1001) / 500.0 - 1.0) * mhd_data_param.deform_sigma());
			}
			transform->SetParameters(trans_param);

			typedef itk::ResampleImageFilter<SliceImageType, SliceImageType> DeformResampleImageFilterType;
			typedef itk::ResampleImageFilter<SliceLabelType, SliceLabelType> DeformResampleLabelFilterType;
			for (int i = 0; i < image_size[2]; ++i) {
				SliceImageType::SizeType size_slice;
				SliceImageType::RegionType region_slice;
				SliceImageType::PointType origin_slice;
				SliceImageType::SpacingType spacing_slice;
				size_slice[0] = image_size[0];
				size_slice[1] = image_size[1];
				region_slice.SetSize(size_slice);
				spacing_slice[0] = image_spacing[0];
				spacing_slice[1] = image_spacing[1];
				origin_slice[0] = image_origin[0];
				origin_slice[1] = image_origin[1];

				SliceImageType::Pointer image_slice = SliceImageType::New();
				image_slice->SetSpacing(spacing_slice);
				image_slice->SetOrigin(origin_slice);
				image_slice->SetRegions(region_slice);
				image_slice->Allocate();
				memcpy(image_slice->GetBufferPointer(), image_buffer + i * slice_pixel_num, slice_pixel_num * sizeof(Dtype));

				DeformResampleImageFilterType::Pointer resample_deform_image = DeformResampleImageFilterType::New();
				resample_deform_image->SetUseReferenceImage(true);
				resample_deform_image->SetTransform(transform);
				resample_deform_image->SetReferenceImage(image_slice);
				resample_deform_image->SetInput(image_slice);
				resample_deform_image->Update();
				SliceImageType::Pointer deform_image_slice = resample_deform_image->GetOutput();
				deform_image_slice->DisconnectPipeline();
				memcpy(image_buffer + i * slice_pixel_num, deform_image_slice->GetBufferPointer(), slice_pixel_num * sizeof(Dtype));

				if (output_labels_) {
					SliceLabelType::Pointer label_slice = SliceLabelType::New();
					label_slice->SetSpacing(spacing_slice);
					label_slice->SetOrigin(origin_slice);
					label_slice->SetRegions(region_slice);
					label_slice->Allocate();
					memcpy(label_slice->GetBufferPointer(), label_buffer + i * slice_pixel_num, slice_pixel_num * sizeof(char));

					DeformResampleLabelFilterType::Pointer resample_deform_label = DeformResampleLabelFilterType::New();
					resample_deform_label->SetUseReferenceImage(true);
					resample_deform_label->SetTransform(transform);
					resample_deform_label->SetReferenceImage(label_slice);
					resample_deform_label->SetInterpolator(DeformInterpolatorType::New());
					resample_deform_label->SetInput(label_slice);
					resample_deform_label->Update();
					SliceLabelType::Pointer deform_label_slice = resample_deform_label->GetOutput();
					deform_label_slice->DisconnectPipeline();
					memcpy(label_buffer + i * slice_pixel_num, deform_label_slice->GetBufferPointer(), slice_pixel_num * sizeof(char));
				}
			}
			image_buffer = image->GetBufferPointer();
			if (output_labels_) 
				label_buffer = label->GetBufferPointer();
		}

		Dtype* image_data = batch->data_.mutable_cpu_data() + batch_id * pixel_num;
		for (int img_index = 0; img_index < pixel_num; ++img_index) {
			image_data[img_index] = image_buffer[img_index];
		}

		Dtype* info_data = batch->info_.mutable_cpu_data() + batch_id * 19;
		info_data[0] = image_size[2];
		info_data[1] = image_size[1];
		info_data[2] = image_size[0];
		info_data[3] = image_spacing[2];
		info_data[4] = image_spacing[1];
		info_data[5] = image_spacing[0];
		info_data[6] = image_origin[2];
		info_data[7] = image_origin[1];
		info_data[8] = image_origin[0];
		info_data[9] = image_record->origin_size_[2];
		info_data[10] = image_record->origin_size_[1];
		info_data[11] = image_record->origin_size_[0];
		info_data[12] = image_record->origin_spacing_[2];
		info_data[13] = image_record->origin_spacing_[1];
		info_data[14] = image_record->origin_spacing_[0];
		info_data[15] = image_record->origin_origin_[2];
		info_data[16] = image_record->origin_origin_[1];
		info_data[17] = image_record->origin_origin_[0];
		info_data[18] = 1;

		if (output_roi_) {
			Dtype* roi_data = batch->roi_.mutable_cpu_data() + batch_id * roi_num * 8;
			for (int roi_id = 0; roi_id < roi_num; ++roi_id) {
				roi_data[roi_id * 8 + 0] = (image_record->roi_x0_[roi_id] * image_record->origin_spacing_[0] + image_record->origin_origin_[0] - image_origin[0]) / image_spacing[0];
				roi_data[roi_id * 8 + 1] = (image_record->roi_y0_[roi_id] * image_record->origin_spacing_[1] + image_record->origin_origin_[1] - image_origin[1]) / image_spacing[1];
				roi_data[roi_id * 8 + 2] = (image_record->roi_z0_[roi_id] * image_record->origin_spacing_[2] + image_record->origin_origin_[2] - image_origin[2]) / image_spacing[2];
				roi_data[roi_id * 8 + 3] = (image_record->roi_x1_[roi_id] * image_record->origin_spacing_[0] + image_record->origin_origin_[0] - image_origin[0]) / image_spacing[0];
				roi_data[roi_id * 8 + 4] = (image_record->roi_y1_[roi_id] * image_record->origin_spacing_[1] + image_record->origin_origin_[1] - image_origin[1]) / image_spacing[1];
				roi_data[roi_id * 8 + 5] = (image_record->roi_z1_[roi_id] * image_record->origin_spacing_[2] + image_record->origin_origin_[2] - image_origin[2]) / image_spacing[2];
				roi_data[roi_id * 8 + 6] = image_record->roi_label_[roi_id];
				roi_data[roi_id * 8 + 7] = (roi_data[roi_id * 8 + 2] >= 0 && roi_data[roi_id * 8 + 5] <= image_size[2] - 1); // disable truncated gt_boxes
			}
		}

		if (output_labels_) {
			Dtype* label_data = batch->label_.mutable_cpu_data() + batch_id * pixel_num;
			for (int img_index = 0; img_index < pixel_num; ++img_index) {
				label_data[img_index] = label_buffer[img_index];
			}
		}
	}

	// go to the next iter
	lines_id_++;
	iter_id_++;
	if (lines_id_ >= lines_size) {
		// We have reached the end. Restart from the first.
		DLOG(INFO) << "Restarting data prefetching from start.";
		lines_id_ = 0;
		if (this->phase_ == TRAIN && this->layer_param_.mhd_data_param().shuffle()) {
			ShuffleImages();
		}
	}
}

INSTANTIATE_CLASS(MHDRoiDataLayer);
REGISTER_LAYER_CLASS(MHDRoiData);

}  // namespace caffe
