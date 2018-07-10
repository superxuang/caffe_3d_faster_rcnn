#include "caffe/fast_rcnn_layers.hpp"

namespace caffe {

	double* GenerateAnchors(const AnchorParameter& anchor_param, int& anchor_num)
	{
		const int width_num = anchor_param.width_size();
		const int height_num = anchor_param.height_size();
		const int length_num = anchor_param.length_size();
		anchor_num = length_num * height_num * width_num;
		double* anchors = new double[anchor_num * 6];
		for (int i = 0; i < length_num; ++i) {
			for (int j = 0; j < height_num; ++j) {
				for (int k = 0; k < width_num; ++k) {
					anchors[(i * height_num * width_num + j * width_num + k) * 6 + 0] = -0.5 * anchor_param.width(k);
					anchors[(i * height_num * width_num + j * width_num + k) * 6 + 1] = -0.5 * anchor_param.height(j);
					anchors[(i * height_num * width_num + j * width_num + k) * 6 + 2] = -0.5 * anchor_param.length(i);
					anchors[(i * height_num * width_num + j * width_num + k) * 6 + 3] = 0.5 * anchor_param.width(k) - 1.0;
					anchors[(i * height_num * width_num + j * width_num + k) * 6 + 4] = 0.5 * anchor_param.height(j) - 1.0;
					anchors[(i * height_num * width_num + j * width_num + k) * 6 + 5] = 0.5 * anchor_param.length(i) - 1.0;
				}
			}
		}
		return anchors;
	}

}  // namespace caffe
