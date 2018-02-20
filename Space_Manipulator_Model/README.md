# Space Manipulator Model
空间机械臂动力学模型，根据SpaceDyn编写
+ 已完成前向动力学
+ 某些变量的存储下标并没有从零开始，待规范合理化
+ 只考虑转动关节，未处理平动关节

## 0. 变量说明（self）：
1. 构型相关：
    1. **connect_lower**: [1 x (n-1)] 低连接，connect_lower[a] = b 表示编号为 *a+1* 连杆的低连接为编号 b 的连杆
    2. **connect_upper**: [(n-1) x (n-1)] 高连接， $ connect\_upper[ij]=\begin{cases}1 & \text{if} i = connect\_lower[j]\\-1 &\text{if} i=j\\0 &otherwise\end{cases}$
    3. **connect_end**: [1 x (n-1)] 若某连杆 *i+1* 与末端相连，则connect_end[i]=1
    4. **connect_base**: [1 x (n-1)] 若某连杆 *i+1* 与基座相连，则connect_base[i]=1
    5. **link_num**:连杆数目 n ,编号从0到n-1，其中编号0的连杆为基座
2. 关节类型：目前只考虑旋转关节，不考虑平动关节
3. 惯量矩阵：**inertia**
4. 质量：**mass**
5. 初始姿态：
    1. **q_from_Bi_to_i**
    2. **orientation_of_end_point**
6. 连杆矢量：**link_vector** 