#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/multi_channel_data_layer.hpp"
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
MultiChannelDataLayer<Dtype>::~MultiChannelDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void MultiChannelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());
  typedef itk::ImageFileReader<ImageType> ImageReaderType;
  typedef itk::ImageFileReader<LabelType> LabelReaderType;

  const MultiChannelDataParameter& mhd_data_param = this->layer_param_.multi_channel_data_param();
  const string& root_folder = mhd_data_param.root_folder();
  const ContourNameList& contour_name_list = mhd_data_param.contour_name_list();
  const ContourWindowLevelList& contour_win_level_list = mhd_data_param.contour_win_level_list();
  const ContourWindowWidthList& contour_win_width_list = mhd_data_param.contour_win_width_list();
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

	  pos1 = pos2;
	  pos2 = line.find_first_of(' ', pos1 + 1);
	  label_range[0][0] = min(label_range[0][0], atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str()));

	  pos1 = pos2;
	  pos2 = line.find_first_of(' ', pos1 + 1);
	  label_range[0][1] = max(label_range[0][1], atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str()));

	  pos1 = pos2;
	  pos2 = line.find_first_of(' ', pos1 + 1);
	  label_range[1][0] = min(label_range[1][0], atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str()));

	  pos1 = pos2;
	  pos2 = line.find_first_of(' ', pos1 + 1);
	  label_range[1][1] = max(label_range[1][1], atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str()));

	  pos1 = pos2;
	  pos2 = line.find_first_of(' ', pos1 + 1);
	  label_range[2][0] = min(label_range[2][0], atoi(line.substr(pos1 + 1, pos2 - pos1 - 1).c_str()));

	  pos1 = pos2;
	  label_range[2][1] = max(label_range[2][1], atoi(line.substr(pos1 + 1).c_str()));

      contour_labels.push_back(label_value);
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
      char* label_final_output_buffer = new char[buffer_length];
      memset(label_final_output_buffer, 0, sizeof(char) * buffer_length);
      for (int i = 0; i < buffer_length; ++i) {
        for (int j = 0; j < contour_labels.size(); ++j) {
          if (label_buffer[i] == contour_labels[j]) {
            label_final_output_buffer[i] = exist_contours[j];
            break;
          }
        }
      }
      memcpy(label_buffer, label_final_output_buffer, sizeof(char) * buffer_length);
      delete[]label_final_output_buffer;
	  
      // resample
	  if (mhd_data_param.resample_volume()) {
        ImageType::SizeType size_src = image->GetBufferedRegion().GetSize();
        ImageType::SpacingType spacing_src = image->GetSpacing();
        ImageType::PointType origin_src = image->GetOrigin();

        ImageType::SizeType size_resample;
		size_resample[0] = mhd_data_param.resample_volume_width();
		size_resample[1] = mhd_data_param.resample_volume_height();
		size_resample[2] = mhd_data_param.resample_volume_length();;
        ImageType::SpacingType spacing_resample;
		spacing_resample[0] = mhd_data_param.resample_spacing_x();
		spacing_resample[1] = mhd_data_param.resample_spacing_y();
		spacing_resample[2] = mhd_data_param.resample_spacing_z();
        ImageType::PointType origin_resample;
		if (mhd_data_param.center_align_label()) {
		  origin_resample[0] = origin_src[0] + ((label_range[0][0] + label_range[0][1]) * spacing_src[0] - size_resample[0] * spacing_resample[0]) * 0.5;
		  origin_resample[1] = origin_src[1] + ((label_range[1][0] + label_range[1][1]) * spacing_src[1] - size_resample[1] * spacing_resample[1]) * 0.5;
		  origin_resample[2] = origin_src[2] + ((label_range[2][0] + label_range[2][1]) * spacing_src[2] - size_resample[2] * spacing_resample[2]) * 0.5;
		}
		else {
		  origin_resample[0] = origin_src[0] + 0.5 * (size_src[0] * spacing_src[0] - size_resample[0] * spacing_resample[0]);
		  origin_resample[1] = origin_src[1] + 0.5 * (size_src[1] * spacing_src[1] - size_resample[1] * spacing_resample[1]);
		  origin_resample[2] = origin_src[2] + 0.5 * (size_src[2] * spacing_src[2] - size_resample[2] * spacing_resample[2]);
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

      lines_.push_back(
        std::make_pair(info_file_name,
        std::make_pair(image, label)));
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
  LOG(INFO) << "A total of " << lines_.size() << " slices.";

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

  const int batch_size = mhd_data_param.batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";

  vector<int> data_shape(5);
  data_shape[0] = batch_size;
  data_shape[1] = contour_win_width_list.win_width_size() + 1;
  data_shape[2] = mhd_data_param.resample_volume_length();
  data_shape[3] = mhd_data_param.resample_volume_height();
  data_shape[4] = mhd_data_param.resample_volume_width();

  vector<int> label_shape(5);
  label_shape[0] = batch_size;
  label_shape[1] = 1;
  label_shape[2] = mhd_data_param.resample_volume_length();
  label_shape[3] = mhd_data_param.resample_volume_height();
  label_shape[4] = mhd_data_param.resample_volume_width();

  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(data_shape);
    this->prefetch_[i].label_.Reshape(label_shape);
  }

  top[0]->Reshape(data_shape);
  top[1]->Reshape(label_shape);

  LOG(INFO) 
    << "input data size: " 
    << top[0]->num() << ","
    << top[0]->channels() << "," 
    << top[0]->shape(2) << ","
    << top[0]->shape(3) << ","
    << top[0]->shape(4);
}

template <typename Dtype>
void MultiChannelDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void MultiChannelDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  typedef itk::ImageFileReader<ImageType> ImageReaderType;
  typedef itk::ImageFileReader<LabelType> LabelReaderType;
  typedef itk::ImageFileWriter<ImageType> ImageWriterType;
  typedef itk::ImageFileWriter<LabelType> LabelWriterType;
  typedef itk::HistogramMatchingImageFilter<ImageType, ImageType> HMFilterType;
  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
  typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
  typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;

  CPUTimer batch_timer;
  batch_timer.Start();
  double hist_time = 0;
  CPUTimer hist_timer;
  double deform_time = 0;
  CPUTimer deform_timer;

  const MultiChannelDataParameter& mhd_data_param = this->layer_param_.multi_channel_data_param();
  const int batch_size = mhd_data_param.batch_size();
  const string& root_folder = mhd_data_param.root_folder();
  const ContourNameList& contour_name_list = mhd_data_param.contour_name_list();
  const ContourWindowLevelList& contour_win_level_list = mhd_data_param.contour_win_level_list();
  const ContourWindowWidthList& contour_win_width_list = mhd_data_param.contour_win_width_list();
  const int lines_size = lines_.size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    CHECK_GT(lines_size, lines_id_);

    ImageType::Pointer image_src = lines_[lines_id_].second.first;
    LabelType::Pointer label_src = lines_[lines_id_].second.second;

	ImageType::RegionType region_src = image_src->GetBufferedRegion();
	ImageType::SizeType size_src = image_src->GetBufferedRegion().GetSize();
    ImageType::SpacingType spacing_src = image_src->GetSpacing();
    ImageType::PointType origin_src = image_src->GetOrigin();
    ImageType::DirectionType direct_src = image_src->GetDirection();

	const int src_buffer_length = image_src->GetBufferedRegion().GetNumberOfPixels();
	const int src_slice_buffer_length = size_src[0] * size_src[1];
    const Dtype* image_src_buffer = image_src->GetBufferPointer();
	const char* label_src_buffer = label_src->GetBufferPointer();

	ImageType::Pointer image_output = ImageType::New();
	image_output->SetRegions(region_src);
	image_output->SetSpacing(spacing_src);
	image_output->SetOrigin(origin_src);
	image_output->SetDirection(direct_src);
	image_output->Allocate();
	Dtype* image_output_buffer = image_output->GetBufferPointer();
	memcpy(image_output_buffer, image_src_buffer, sizeof(Dtype) * src_buffer_length);

	LabelType::Pointer label_output = LabelType::New();
	label_output->SetRegions(region_src);
	label_output->SetSpacing(spacing_src);
	label_output->SetOrigin(origin_src);
	label_output->SetDirection(direct_src);
	label_output->Allocate();
	char* label_output_buffer = label_output->GetBufferPointer();
	memcpy(label_output_buffer, label_src_buffer, sizeof(char) * src_buffer_length);

	// histogram matching
	if (this->phase_ == TRAIN && mhd_data_param.hist_matching()) {
      hist_timer.Start();

      caffe::rng_t* rng = static_cast<caffe::rng_t*>(ind_rng_->generator());
      const int lines_id_next = (*rng)() % lines_size;
      ImageType::Pointer image_hist = lines_[lines_id_next].second.first;

      //ImageWriterType::Pointer writer = ImageWriterType::New();
      //writer->SetInput(image_stack);
      //writer->SetFileName("F:/Deep/Output_Image_Src.mhd");
      //writer->Update();
      //writer = ImageWriterType::New();
      //writer->SetInput(image_hist_slice);
      //writer->SetFileName("F:/Deep/Output_Image_Hist.mhd");
      //writer->Update();
      
      HMFilterType::Pointer hist_matching = HMFilterType::New();
      hist_matching->SetReferenceImage(image_hist);
      hist_matching->SetInput(image_src);
	  hist_matching->SetNumberOfHistogramLevels(80);
	  hist_matching->SetNumberOfMatchPoints(12);
	  hist_matching->ThresholdAtMeanIntensityOn();
	  hist_matching->Update();
	  image_output = hist_matching->GetOutput();

      //writer = ImageWriterType::New();
      //writer->SetInput(image_stack);
      //writer->SetFileName("F:/Deep/Output_Image_Src_Hist.mhd");
      //writer->Update();

      hist_time += hist_timer.MicroSeconds();
    }
    
	image_output_buffer = image_output->GetBufferPointer();
	label_output_buffer = label_output->GetBufferPointer();

    // deform stack
	caffe::rng_t* rng = static_cast<caffe::rng_t*>(trans_rng_->generator());
	if (this->phase_ == TRAIN && mhd_data_param.random_deform() > 0 &&
      ((*rng)() % 1001) / 1000.0 >= 1 - mhd_data_param.random_deform()) {

      deform_timer.Start();

      typedef itk::Image<Dtype, 2> SliceImageType;
      typedef itk::Image<char, 2> SliceLabelType;
      SliceImageType::Pointer image_stack_slice = SliceImageType::New();
      SliceImageType::SizeType size_slice;
      SliceImageType::RegionType region_slice;
      SliceImageType::PointType origin_slice;
      SliceImageType::SpacingType spacing_slice;
	  size_slice[0] = size_src[0];
	  size_slice[1] = size_src[1];
      region_slice.SetSize(size_slice);
      spacing_slice[0] = spacing_src[0];
	  spacing_slice[1] = spacing_src[1];
      origin_slice[0] = origin_src[0];
	  origin_slice[1] = origin_src[1];
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
	    
	  //LabelWriterType::Pointer writer_label = LabelWriterType::New();
	  //writer_label->SetInput(label_output);
	  //writer_label->SetFileName("F:/Deep/Label_before_deform.mhd");
	  //writer_label->Update();
	  //ImageWriterType::Pointer writer_image = ImageWriterType::New();
	  //writer_image->SetInput(image_output);
	  //writer_image->SetFileName("F:/Deep/Image_before_deform.mhd");
	  //writer_image->Update();
	    
      typedef itk::ResampleImageFilter<SliceImageType, SliceImageType> DeformResampleImageFilterType;
      typedef itk::ResampleImageFilter<SliceLabelType, SliceLabelType> DeformResampleLabelFilterType;
	  for (int i = 0; i < size_src[2]; ++i) {
        SliceImageType::SizeType size_slice;
        SliceImageType::RegionType region_slice;
        SliceImageType::PointType origin_slice;
        SliceImageType::SpacingType spacing_slice;
        size_slice[0] = size_src[0];
		size_slice[1] = size_src[1];
        region_slice.SetSize(size_slice);
        spacing_slice[0] = spacing_src[0];
        spacing_slice[1] = spacing_src[1];
        origin_slice[0] = origin_src[0];
        origin_slice[1] = origin_src[1];

        SliceImageType::Pointer image_slice = SliceImageType::New();
        image_slice->SetSpacing(spacing_slice);
        image_slice->SetOrigin(origin_slice);
        image_slice->SetRegions(region_slice);
        image_slice->Allocate();
		memcpy(image_slice->GetBufferPointer(), image_output_buffer + i * src_slice_buffer_length, src_slice_buffer_length * sizeof(Dtype));

        DeformResampleImageFilterType::Pointer resample_deform_image = DeformResampleImageFilterType::New();
        resample_deform_image->SetUseReferenceImage(true);
        resample_deform_image->SetTransform(transform);
        resample_deform_image->SetReferenceImage(image_slice);
        resample_deform_image->SetInput(image_slice);
        resample_deform_image->Update();
        SliceImageType::Pointer deform_image_slice = resample_deform_image->GetOutput();
        deform_image_slice->DisconnectPipeline();
		memcpy(image_output_buffer + i * src_slice_buffer_length, deform_image_slice->GetBufferPointer(), src_slice_buffer_length * sizeof(Dtype));

        SliceLabelType::Pointer label_slice = SliceLabelType::New();
        label_slice->SetSpacing(spacing_slice);
        label_slice->SetOrigin(origin_slice);
        label_slice->SetRegions(region_slice);
        label_slice->Allocate();
		memcpy(label_slice->GetBufferPointer(), label_output_buffer + i * src_slice_buffer_length, src_slice_buffer_length * sizeof(char));
		
		DeformResampleLabelFilterType::Pointer resample_deform_label = DeformResampleLabelFilterType::New();
        resample_deform_label->SetUseReferenceImage(true);
        resample_deform_label->SetTransform(transform);
        resample_deform_label->SetReferenceImage(label_slice);
        resample_deform_label->SetInterpolator(DeformInterpolatorType::New());
        resample_deform_label->SetInput(label_slice);
        resample_deform_label->Update();
        SliceLabelType::Pointer deform_label_slice = resample_deform_label->GetOutput();
        deform_label_slice->DisconnectPipeline();
		memcpy(label_output_buffer + i * src_slice_buffer_length, deform_label_slice->GetBufferPointer(), src_slice_buffer_length * sizeof(char));
      }
      
	  //writer_label = LabelWriterType::New();
	  //writer_label->SetInput(label_output);
	  //writer_label->SetFileName("F:/Deep/Label_after_deform.mhd");
	  //writer_label->Update();
	  //writer_image = ImageWriterType::New();
	  //writer_image->SetInput(image_output);
	  //writer_image->SetFileName("F:/Deep/Image_after_deform.mhd");
	  //writer_image->Update();

      deform_time += deform_timer.MicroSeconds();
    }

    image_output_buffer = image_output->GetBufferPointer();
	label_output_buffer = label_output->GetBufferPointer();

	//LabelWriterType::Pointer writer_label = LabelWriterType::New();
	//writer_label->SetInput(label_output);
	//writer_label->SetFileName("F:/Output_Label.mhd");
	//writer_label->Update();
	//ImageWriterType::Pointer writer_image = ImageWriterType::New();
	//writer_image->SetInput(image_output);
	//writer_image->SetFileName("F:/Output_Image.mhd");
	//writer_image->Update();

    if (item_id == 0) {
      vector<int> data_shape(5);
      data_shape[0] = batch_size;
	  data_shape[1] = contour_win_width_list.win_width_size() + 1;
	  data_shape[2] = size_src[2];
      data_shape[3] = size_src[1];
	  data_shape[4] = size_src[0];
      batch->data_.Reshape(data_shape);
      vector<int> label_shape(5);
      label_shape[0] = batch_size;
      label_shape[1] = 1;
	  label_shape[2] = size_src[2];
	  label_shape[3] = size_src[1];
	  label_shape[4] = size_src[0];
      batch->label_.Reshape(label_shape);
    }

    Dtype* prefetch_data = batch->data_.mutable_cpu_data();
    Dtype* prefetch_label = batch->label_.mutable_cpu_data();
    std::vector<int> offset_inds(5);
    offset_inds[0] = item_id;
    offset_inds[1] = 0;
    offset_inds[2] = 0;
    offset_inds[3] = 0;
    offset_inds[4] = 0;
    Dtype* image_data = prefetch_data + batch->data_.offset(offset_inds);
    Dtype* label_data = prefetch_label + batch->label_.offset(offset_inds);

	for (int img_index = 0; img_index < src_buffer_length; ++img_index) {
      label_data[img_index] = label_output_buffer[img_index];
    }

    for (int chn = 0; chn < contour_win_width_list.win_width_size(); ++chn) {
      offset_inds[1] = chn + 1;
      image_data = prefetch_data + batch->data_.offset(offset_inds);
      float win_level = contour_win_level_list.win_level(chn);
      float win_width = contour_win_width_list.win_width(chn);
      for (int img_index = 0; img_index < src_buffer_length; ++img_index) {
        image_data[img_index] = exp(-(image_output_buffer[img_index] - win_level) * (image_output_buffer[img_index] - win_level) / (win_width * win_width));
      }
    }

    // rescale image stack to [0.0, 1.0]
    Dtype buff_max = image_output_buffer[0];
    Dtype buff_min = image_output_buffer[0];
    for (int i = 0; i < src_buffer_length; ++i) {
      if (buff_max < image_output_buffer[i])
        buff_max = image_output_buffer[i];
      if (buff_min > image_output_buffer[i])
        buff_min = image_output_buffer[i];
    }
    double buff_scale = (buff_max > buff_min) ? 1.0 / (buff_max - buff_min) : 0.0;
    offset_inds[1] = 0;
    image_data = prefetch_data + batch->data_.offset(offset_inds);
    for (int img_index = 0; img_index < src_buffer_length; ++img_index) {
      image_data[img_index] = (image_output_buffer[img_index] - buff_min) * buff_scale;
    }
    
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
	    if (this->phase_ == TRAIN && mhd_data_param.shuffle()) {
        ShuffleImages();
      }
    }

	//typedef itk::Image<Dtype, 3> ImageType;
	//typedef itk::ImageFileWriter<ImageType> ImageWriterType;
	//ImageType::Pointer output_image = ImageType::New();
	//ImageType::Pointer output_prob = ImageType::New();
	//ImageType::SizeType size_output;
	//ImageType::RegionType region_output;
	//ImageType::PointType origin_output;
	//ImageType::SpacingType spacing_output;
	//size_output[0] = size_src[0];
	//size_output[1] = size_src[1];
	//size_output[2] = size_src[2];
	//region_output.SetSize(size_output);
	//spacing_output[0] = 1;
	//spacing_output[1] = 1;
	//spacing_output[2] = 1;
	//origin_output[0] = 0;
	//origin_output[1] = 0;
	//origin_output[2] = 0;
	//output_image->SetSpacing(spacing_output);
	//output_image->SetOrigin(origin_output);
	//output_image->SetRegions(region_output);
	//output_image->Allocate();
	//output_prob->SetSpacing(spacing_output);
	//output_prob->SetOrigin(origin_output);
	//output_prob->SetRegions(region_output);
	//output_prob->Allocate();
	//int pixel_num_1 = size_output[0] * size_output[1] * size_output[2];
	//{
	//	memcpy(output_image->GetBufferPointer(), image_data, sizeof(Dtype) * pixel_num_1);
	//	memcpy(output_prob->GetBufferPointer(), image_data + pixel_num_1, sizeof(Dtype) * pixel_num_1);
	//	ImageWriterType::Pointer writer_image = ImageWriterType::New();
	//	writer_image->SetInput(output_image);
	//	writer_image->SetFileName("F:/channel_0.mhd");
	//	writer_image->Update();
	//	ImageWriterType::Pointer writer_prob = ImageWriterType::New();
	//	writer_prob->SetInput(output_prob);
	//	writer_prob->SetFileName("F:/channel_1.mhd");
	//	writer_prob->Update();
	//}
	//{
	//	memcpy(output_image->GetBufferPointer(), image_data + 2 * pixel_num_1, sizeof(Dtype) * pixel_num_1);
	//	memcpy(output_prob->GetBufferPointer(), image_data + 3 * pixel_num_1, sizeof(Dtype) * pixel_num_1);
	//	ImageWriterType::Pointer writer_image = ImageWriterType::New();
	//	writer_image->SetInput(output_image);
	//	writer_image->SetFileName("F:/channel_2.mhd");
	//	writer_image->Update();
	//	ImageWriterType::Pointer writer_prob = ImageWriterType::New();
	//	writer_prob->SetInput(output_prob);
	//	writer_prob->SetFileName("F:/channel_3.mhd");
	//	writer_prob->Update();
	//}
  }
  batch_timer.Stop();
  if (this->phase_ == TRAIN) {
	//LOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	//LOG(INFO) << "Histogram time: " << hist_time / 1000 << " ms.";
	//LOG(INFO) << "   Deform time: " << deform_time / 1000 << " ms.";
  }
}

INSTANTIATE_CLASS(MultiChannelDataLayer);
REGISTER_LAYER_CLASS(MultiChannelData);

}  // namespace caffe
