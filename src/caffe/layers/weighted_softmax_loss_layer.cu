#include <algorithm>
#include <cfloat>
#include <vector>
#include <glog/logging.h>

#include "caffe/layer.hpp"
#include "caffe/layers/weighted_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {  

template <typename Dtype>
__global__ void CalculateChannelWeights(const int nthreads,
          const Dtype* label, const int num, const int channel_dim, 
		  const int spatial_dim, const bool has_ignore_label_, const int ignore_label_,
          int* weights) {
  extern __shared__ int tmp_weights[];
  if (threadIdx.x < num * channel_dim)
	tmp_weights[threadIdx.x] = 0;
  __syncthreads();
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (!has_ignore_label_ || label_value != ignore_label_) {
	  atomicAdd(&(tmp_weights[n * channel_dim + label_value]), 1);
    }
  }
  __syncthreads();
  if (threadIdx.x < num * channel_dim)
    atomicAdd(&(weights[threadIdx.x]), tmp_weights[threadIdx.x]);
}

template <typename Dtype>
__global__ void WeightedSoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
		  const Dtype* weights_data, 
		  const int num, const int dim, 
		  const int spatial_dim, const bool has_ignore_label_, 
		  const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      Dtype pt = prob_data[n * dim + label_value * spatial_dim + s];
	  loss[index] = -weights_data[n * channels + label_value] * log(max(pt, Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void WeightedFocalLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
		  const Dtype gamma, const Dtype* weights_data, 
		  const int num, const int dim, 
		  const int spatial_dim, const bool has_ignore_label_, 
		  const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      Dtype pt = prob_data[n * dim + label_value * spatial_dim + s];
	  loss[index] = -weights_data[n * channels + label_value] * powf(1 - pt, gamma) * log(max(pt, Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void FocalLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
		  const Dtype alpha, const Dtype gamma, 
		  const int num, const int dim,
		  const int spatial_dim, const bool has_ignore_label_, 
		  const int ignore_label_, Dtype* counts) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      Dtype pt = prob_data[n * dim + label_value * spatial_dim + s];
	  loss[index] = -alpha * powf(1 - pt, gamma) * log(max(pt, Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();

  if (method_ == WeightedSoftmaxLossParameter_Method_ADAPTIVE_WEIGHT || method_ == WeightedSoftmaxLossParameter_Method_ADAPTIVE_FOCAL_LOSS) {
    Blob<int> tmp_weights;
    std::vector<int> weights_shape(2);
    weights_shape[0] = outer_num_;
    weights_shape[1] = channel_num_;
    tmp_weights.Reshape(weights_shape);
    caffe_memset(outer_num_ * channel_num_ * sizeof(int), 0, tmp_weights.mutable_cpu_data());
    int* tmp_weights_gpu_data = tmp_weights.mutable_gpu_data();
    CalculateChannelWeights<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS, outer_num_ * channel_num_ * sizeof(Dtype)>>>(nthreads, label, outer_num_, 
        channel_num_, inner_num_, has_ignore_label_, ignore_label_, tmp_weights_gpu_data);
    int* tmp_weights_cpu_data = tmp_weights.mutable_cpu_data();
    Dtype* weights_cpu_data = channel_weights_.mutable_cpu_data();
    for (int i = 0; i < outer_num_; ++i) {
      Dtype max_weight = 0;
      for (int j = 0; j < channel_num_; ++j) {
        if (tmp_weights_cpu_data[i * channel_num_ + j] > max_weight) {
      	  max_weight = tmp_weights_cpu_data[i * channel_num_ + j];
        }
        //LOG(INFO) << "weights_data = " << tmp_weights_cpu_data[i * channel_num_ + j];
      }
      //LOG(INFO) << "max_weight = " << max_weight;
      for (int j = 0; j < channel_num_; ++j) {
        weights_cpu_data[i * channel_num_ + j] = 1;
        if (tmp_weights_cpu_data[i * channel_num_ + j] > 0)
          weights_cpu_data[i * channel_num_ + j] = max_weight / tmp_weights_cpu_data[i * channel_num_ + j];
        else
          weights_cpu_data[i * channel_num_ + j] = 0;
        //LOG(INFO) << "normalized_weights_data = " << weights_cpu_data[i * channel_num_ + j];
      }
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    Dtype* weights_gpu_data = channel_weights_.mutable_gpu_data();
	if (method_ == WeightedSoftmaxLossParameter_Method_ADAPTIVE_WEIGHT) {
      WeightedSoftmaxLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data, weights_gpu_data, 
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
    } else {
      WeightedFocalLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data, gamma_, weights_gpu_data, 
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
    }
  } else {
    FocalLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS >> >(nthreads, prob_data, label, loss_data, alpha_, gamma_, 
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  }
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                        valid_count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void WeightedSoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, const Dtype* weights_data, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
      for (int c = 0; c < channels; ++c) {  
        bottom_diff[n * dim + c * spatial_dim + s] *= weights_data[n * channels + label_value];  
      }
    }
  }
}

template <typename Dtype>
__global__ void WeightedFocalLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, const Dtype gamma, const Dtype* weights_data, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      Dtype pt = bottom_diff[n * dim + label_value * spatial_dim + s];
      for (int c = 0; c < channels; ++c) {
        if (c == label_value){
          bottom_diff[n * dim + c * spatial_dim + s] = powf(1 - pt, gamma) * (gamma * pt * log(max(pt, Dtype(FLT_MIN))) + pt - 1);
        }
        else{
          Dtype pc = bottom_diff[n * dim + c * spatial_dim + s];
          bottom_diff[n * dim + c * spatial_dim + s] = (powf(1 - pt, gamma - 1) * (-gamma * log(max(pt, Dtype(FLT_MIN))) * pt * pc) + powf(1 - pt, gamma) * pc);
        }
      }
      counts[index] = 1;
	  for (int c = 0; c < channels; ++c) {  
        bottom_diff[n * dim + c * spatial_dim + s] *= weights_data[n * channels + label_value];  
      }
    }
  }
}

template <typename Dtype>
__global__ void FocalLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, const Dtype alpha, const Dtype gamma, 
		  Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      Dtype pt = bottom_diff[n * dim + label_value * spatial_dim + s];
      for (int c = 0; c < channels; ++c) {
        if (c == label_value){
          bottom_diff[n * dim + c * spatial_dim + s] = alpha * powf(1 - pt, gamma) * (gamma * pt * log(max(pt, Dtype(FLT_MIN))) + pt - 1);
        }
        else{
          Dtype pc = bottom_diff[n * dim + c * spatial_dim + s];
          bottom_diff[n * dim + c * spatial_dim + s] = alpha * (powf(1 - pt, gamma - 1) * (-gamma * log(max(pt, Dtype(FLT_MIN))) * pt * pc) + powf(1 - pt, gamma) * pc);
        }
      }
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
	if (method_ == WeightedSoftmaxLossParameter_Method_ADAPTIVE_WEIGHT) {
      const Dtype* weights_data = channel_weights_.gpu_data();
  	  WeightedSoftmaxLossBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, weights_data, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
	}
	else if (method_ == WeightedSoftmaxLossParameter_Method_ADAPTIVE_FOCAL_LOSS) {
      const Dtype* weights_data = channel_weights_.gpu_data();
  	  WeightedFocalLossBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, gamma_, weights_data, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
	}
	else {
      FocalLossBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS >> >(nthreads, top_data, label, alpha_, gamma_, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
    }

    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0] /
                              get_normalizer(normalization_, valid_count);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightedSoftmaxWithLossLayer);

}  // namespace caffe
