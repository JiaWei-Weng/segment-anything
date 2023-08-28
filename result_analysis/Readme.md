# mask结果处理代码

这个Python脚本用于对mask结果进行聚类。它可以读取CSV文件，提取图像，计算图像之间的重叠。

## 使用方法

1. **安装依赖**：确保你的系统上安装了Python和以下库：
   - OpenCV (`cv2`)
   - NumPy (`numpy`)
   - pandas (`pandas`)
   - scikit-learn (`sklearn`)

2. **下载脚本**：将本脚本保存到你的计算机上，并记下它的路径。

3. **命令行参数**：你可以在命令行中使用以下参数来运行脚本：

   - `--csv-folder`：CSV文件的文件夹路径，包含了需要处理的元数据文件。例如：
     ```
     python merge.py --csv-folder "H:\3Ddom\segment_anything_backups\data_output\101ND750\images_3_samauto_1"
     ```

   - `--try-num`：可选参数，指定要处理的运行次数，默认为1。例如，如果你想处理10个运行，你可以这样运行脚本：
     ```
     python merge.py --csv-folder "H:\3Ddom\segment_anything_backups\data_output\101ND750\images_3_samauto_1" --try-num 10
     ```

4. **输出结果**：脚本将会生成一个名为`result_mask`的文件夹，其中包含了处理后的图像和其他输出结果。

## 注意事项

- 请确保你的CSV文件的格式与脚本中使用的格式相匹配，以避免出错。
- 请确保路径和文件名的正确性，根据你的目录结构进行相应的调整。

如果你有任何问题或需要更多帮助，可以联系[22112076@zju.edu.cn]。

祝使用愉快！
