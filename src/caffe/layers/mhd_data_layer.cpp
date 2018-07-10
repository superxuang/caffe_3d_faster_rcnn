#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/mhd_data_layer.hpp"
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
MHDDataLayer<Dtype>::~MHDDataLayer<Dtype>() {
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
void MHDDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
  while (std::getline(infile, line)) {
	pos1 = line.find_first_of(' ');
	pos2 = line.find_last_of(' ');
    string image_file_name = line.substr(0, pos1);
    string label_file_name = line.substr(pos1 + 1, pos2 - pos1 - 1);
    string info_file_name = line.substr(pos2 + 1);
    std::ifstream infile_info(root_folder + info_file_name);
    std::vector<int> contour_labels;
    std::vector<int> exist_contours;
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
    }
    if (!contour_labels.empty()) {
      ImageType::DirectionType direct_src;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          direct_src[i][j] = (i == j) ? 1 : 0;
        }
      }

	  ImageRecord* image_record = new ImageRecord;

      ImageReaderType::Pointer reader_image = ImageReaderType::New();
      reader_image->SetFileName(root_folder + image_file_name);
      reader_image->Update();
      ImageType::Pointer image = reader_image->GetOutput();
      image->SetDirection(direct_src);

      LabelReaderType::Pointer reader_label = LabelReaderType::New();
      reader_label->SetFileName(root_folder + label_file_name);
      reader_label->Update();
      LabelType::Pointer label = reader_label->GetOutput();
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

      // resample
	  {
        ImageType::SizeType size_src = image->GetBufferedRegion().GetSize();
        ImageType::SpacingType spacing_src = image->GetSpacing();
        ImageType::PointType origin_src = image->GetOrigin();

        ImageType::SizeType size_resample;
		size_resample[0] = size_src[0] / mhd_data_param.shrink_factor_x();
		size_resample[1] = size_src[1] / mhd_data_param.shrink_factor_y();
		size_resample[2] = size_src[2] / mhd_data_param.shrink_factor_z();
        ImageType::SpacingType spacing_resample;
		spacing_resample[0] = spacing_src[0] * mhd_data_param.shrink_factor_x();
		spacing_resample[1] = spacing_src[1] * mhd_data_param.shrink_factor_y();
		spacing_resample[2] = spacing_src[2] * mhd_data_param.shrink_factor_z();
		ImageType::PointType origin_resample;
		origin_resample[0] = origin_src[0];
		origin_resample[1] = origin_src[1];
		origin_resample[2] = origin_src[2];

		if (size_resample[0] > mhd_data_param.max_width()) {
		  size_resample[0] = mhd_data_param.max_width();
		  origin_resample[0] = origin_src[0] + 0.5 * size_src[0] * spacing_src[0] - 0.5 * size_resample[0] * spacing_resample[0];
		}
		if (size_resample[1] > mhd_data_param.max_height()) {
		  size_resample[1] = mhd_data_param.max_height();
		  origin_resample[1] = origin_src[1] + 0.5 * size_src[1] * spacing_src[1] - 0.5 * size_resample[1] * spacing_resample[1];
		}
		if (size_resample[2] > mhd_data_param.max_length()) {
		  size_resample[2] = mhd_data_param.max_length();
		  origin_resample[2] = origin_src[2] + 0.5 * size_src[2] * spacing_src[2] - 0.5 * size_resample[2] * spacing_resample[2];
		}

		for (int i = 0; i < 3; ++i)	{
		  image_record->origin_size_[i] = size_src[i];
		  image_record->origin_spacing_[i] = spacing_src[i];
		  image_record->origin_origin_[i] = origin_src[i];
		  image_record->size_[i] = size_resample[i];
		  image_record->spacing_[i] = spacing_resample[i];
		  image_record->origin_[i] = origin_resample[i];
		}

        typedef itk::IdentityTransform<double, 3> IdentityTransformType;
        typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
        typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
        typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;

        ResampleImageFilterType::Pointer resampler_image = ResampleImageFilterType::New();
        resampler_image->SetInput(image);
        resampler_image->SetSize(size_resample);
        resampler_image->SetOutputSpacing(spacing_resample);
        resampler_image->SetOutputOrigin(origin_resample);
        resampler_image->SetTransform(IdentityTransformType::New());
        resampler_image->Update();
        image = resampler_image->GetOutput();

        ResampleLabelFilterType::Pointer resampler_label = ResampleLabelFilterType::New();
        resampler_label->SetInput(label);
        resampler_label->SetSize(size_resample);
        resampler_label->SetOutputSpacing(spacing_resample);
        resampler_label->SetOutputOrigin(origin_resample);
        resampler_label->SetTransform(IdentityTransformType::New());
        resampler_label->SetInterpolator(InterpolatorType::New());
        resampler_label->Update();
        label = resampler_label->GetOutput();
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
  // Check if we would need to randomly skip a few data points
  if (mhd_data_param.rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
      mhd_data_param.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  const unsigned int ind_rng_seed = caffe_rng_rand();
  ind_rng_.reset(new Caffe::RNG(ind_rng_seed));
  const unsigned int trans_rng_seed = caffe_rng_rand();
  trans_rng_.reset(new Caffe::RNG(trans_rng_seed));

  vector<int> data_shape(5, 1);
  data_shape[0] = 1;
  data_shape[1] = 1;
  data_shape[2] = 64;
  data_shape[3] = 128;
  data_shape[4] = 128;
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
	this->prefetch_[i].data_.Reshape(data_shape);
	this->prefetch_[i].label_.Reshape(data_shape);
  }

  top[0]->Reshape(data_shape);
  top[1]->Reshape(data_shape);
}

template <typename Dtype>
void MHDDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void MHDDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  typedef itk::ImageFileReader<ImageType> ImageReaderType;
  typedef itk::ImageFileReader<LabelType> LabelReaderType;
  typedef itk::ImageFileWriter<ImageType> ImageWriterType;
  typedef itk::ImageFileWriter<LabelType> LabelWriterType;
  typedef itk::HistogramMatchingImageFilter<ImageType, ImageType> HMFilterType;
  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
  typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
  typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;

  const MHDDataParameter& mhd_data_param = this->layer_param_.mhd_data_param();
  const int batch_size = mhd_data_param.batch_size();
  const string& root_folder = mhd_data_param.root_folder();
  const ContourNameList& contour_name_list = mhd_data_param.contour_name_list();
  const int lines_size = lines_.size();
  CHECK_GT(lines_size, lines_id_);
  const ImageRecord* image_record = lines_[lines_id_];
  
  ImageType::Pointer src_image = image_record->image_;
  LabelType::Pointer src_label = image_record->label_;

  ImageType::RegionType image_region = src_image->GetBufferedRegion();
  ImageType::SizeType image_size = src_image->GetBufferedRegion().GetSize();
  ImageType::SpacingType image_spacing = src_image->GetSpacing();
  ImageType::PointType image_origin = src_image->GetOrigin();
  ImageType::DirectionType image_direct = src_image->GetDirection();

  const int pixel_num = src_image->GetBufferedRegion().GetNumberOfPixels();
  const int slice_pixel_num = image_size[0] * image_size[1];
  const Dtype* src_image_buffer = src_image->GetBufferPointer();
  const char* src_label_buffer = src_label->GetBufferPointer();

  ImageType::Pointer image = ImageType::New();
  image->SetRegions(image_region);
  image->SetSpacing(image_spacing);
  image->SetOrigin(image_origin);
  image->SetDirection(image_direct);
  image->Allocate();
  Dtype* image_buffer = image->GetBufferPointer();
  memcpy(image_buffer, src_image_buffer, sizeof(Dtype) * pixel_num);

  LabelType::Pointer label = LabelType::New();
  label->SetRegions(image_region);
  label->SetSpacing(image_spacing);
  label->SetOrigin(image_origin);
  label->SetDirection(image_direct);
  label->Allocate();
  char* label_buffer = label->GetBufferPointer();
  memcpy(label_buffer, src_label_buffer, sizeof(char) * pixel_num);

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
  }
    
  image_buffer = image->GetBufferPointer();
  label_buffer = label->GetBufferPointer();

  // rescale image stack to [0.0, 1.0]
  Dtype buff_max = image_buffer[0];
  Dtype buff_min = image_buffer[0];
  for (int i = 0; i < pixel_num; ++i) {
    if (buff_max < image_buffer[i])
      buff_max = image_buffer[i];
	if (buff_min > image_buffer[i])
      buff_min = image_buffer[i];
  }
  double buff_scale = (buff_max > buff_min) ? 1.0 / (buff_max - buff_min) : 0.0;
  for (int i = 0; i < pixel_num; ++i) {
    image_buffer[i] = (image_buffer[i] - buff_min) * buff_scale;
  }

  // deform stack
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
  label_buffer = label->GetBufferPointer();

  vector<int> data_shape(5);
  data_shape[0] = 1;
  data_shape[1] = 1;
  data_shape[2] = image_size[2];
  data_shape[3] = image_size[1];
  data_shape[4] = image_size[0];
  batch->data_.Reshape(data_shape);
  Dtype* image_data = batch->data_.mutable_cpu_data();
  for (int img_index = 0; img_index < pixel_num; ++img_index) {
    image_data[img_index] = image_buffer[img_index];
  }

  if (this->phase_ == TRAIN) {
    batch->label_.Reshape(data_shape);
    Dtype* label_data = batch->label_.mutable_cpu_data();
    for (int img_index = 0; img_index < pixel_num; ++img_index) {
      label_data[img_index] = label_buffer[img_index];
    }
  }
  else {
	std::vector<int> data_info_shape(2);
	data_info_shape[0] = 1;
	data_info_shape[1] = 18;
	batch->label_.Reshape(data_info_shape);
	Dtype* label_data = batch->label_.mutable_cpu_data();
	for (int i = 0; i < 3; ++i) {
      label_data[0 + i] = image_record->origin_size_[i];
      label_data[3 + i] = image_record->origin_spacing_[i];
	  label_data[6 + i] = image_record->origin_origin_[i];
	  label_data[9 + i] = image_record->size_[i];
      label_data[12 + i] = image_record->spacing_[i];
      label_data[15 + i] = image_record->origin_[i];
	}
  }
  
  // go to the next iter
  lines_id_++;
  if (lines_id_ >= lines_size) {
    // We have reached the end. Restart from the first.
    DLOG(INFO) << "Restarting data prefetching from start.";
    lines_id_ = 0;
	if (this->phase_ == TRAIN && this->layer_param_.mhd_data_param().shuffle()) {
      ShuffleImages();
    }
  }
}

INSTANTIATE_CLASS(MHDDataLayer);
REGISTER_LAYER_CLASS(MHDData);

}  // namespace caffe
