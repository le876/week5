# TuRBO �Ż��㷨��Ŀ

## ��Ŀ����

����Ŀ�ǻ��� NeurIPS 2019 ���� ***Scalable Global Optimization via Local Bayesian Optimization*** �� TuRBO �㷨ʵ�֡�ͨ��TuRBO5�ܹ�ʵ��һ����20άSchwefel������20άrosenbrock������800��ѵ������200����֤����������TuRBO5�Ż����ҵ���Ӧ������ȫ����Сֵ��


## ��Ŀ�ṹ

```
TuRBO/
������ turbo/                    # �����㷨ʵ��
��   ������ turbo_m.py           # TuRBO-m �����������㷨
��   ������ turbo_1.py           # TuRBO-1 �����������㷨
��   ������ gp.py                # Gaussian Process ģ��
��   ������ utils.py             # ���ߺ���
������ data/                     # ѵ������
��   ������ Schwefel_x_train.npy    # Schwefel ����ѵ������
��   ������ Schwefel_y_train.npy    # Schwefel ����ѵ�����
��   ������ Rosenbrock_x_train.npy  # Rosenbrock ����ѵ������
��   ������ Rosenbrock_y_train.npy  # Rosenbrock ����ѵ�����
������ results/                  # ������Ŀ¼
������ main.py                  # ��ִ�нű�
������ functions.py             # Ŀ�꺯������
������ plotting.py              # ���ӻ�����
������ config.py                # ���ù���
������ requirements.txt         # �����б�
```
## ���ٿ�ʼ

### ����Ҫ��

```bash
# �Ƽ�ʹ�� conda ����
conda create -n turbo_env python=3.10
conda activate turbo_env

# ��װ����
pip install -r requirements.txt
```

### ����ʹ��

```bash
# Schwefel �����Ż������Զ��� Trust Region ������
python main.py --task schwefel --mode advanced

# Rosenbrock �����Ż����������任���Զ��������
python main.py --task rosenbrock --mode advanced
```

#### �Զ������

```bash
python main.py --task schwefel --mode simple \
    --dim 20 \
    --n_trust_regions 5 \
    --batch_size_per_tr 4 \
    --num_iterations 3 \
    --device cpu \
    --verbose True
```

## Ŀ�꺯��

### Schwefel ����
- **ά��**: 20 ά
- **������**: [-500, 500]^20
- **������ʽ**: f(x) = 418.9829 �� d - ��[x_i �� sin(��|x_i|)]
- **ȫ������ֵ**: Լ 0 (�� x_i �� 420.9687 ��)
- **�ص�**: ���д����ֲ����Ž⣬�ǲ���ȫ���Ż��㷨�ľ��亯��

### Rosenbrock ����
- **ά��**: 20 ά
- **������**: [-2.048, 2.048]^20
- **������ʽ**: f(x) = ��[100(x_{i+1} - x_i?)? + (1 - x_i)?]
- **ȫ������ֵ**: 0 (�� x_i = 1 ��)
- **�ص�**: ��խ������ɽ�ȣ��������ѣ��ʺϲ����㷨�ľ�ϸ��������

### ���ӻ�����
- **��������**: ��ʾ�Ż������е����ֵ�仯
- **GP Ԥ������**: ���� Gaussian Process ģ�͵��������
- **���������ݻ�**: ���ӻ�����������Ĵ�С��λ�ñ仯
- **��������ʷ**: ���� GP ģ�ͳ������ı仯����
- **̽��·��**: ��ʾ�㷨�������켣�ͺ�ѡ��ֲ�

## ����ϸ��

### TuRBO �㷨ԭ��

TuRBO ��һ�ֻ��� **������Trust Region��** �� **��Ҷ˹�Ż���Bayesian Optimization, BO��** ��ȫ���Ż��㷨��ּ�ڽ����ά�����Ӻںк������Ż����⡣�����˼���ǣ�

- **�ֲ���ģ** ��ͨ������ֲ���˹���̣�Gaussian Process, GP��ģ�����ȫ��ģ�ͣ����ٸ�ά�ռ�Ľ�ģ���Ӷȡ�
- **��̬������** ���ھֲ������ڵ���������Χ�������򣩣�ƽ��̽����Exploration�������ã�Exploitation��������ȫ���Ż��еĹ���̽�����⡣
- **����ϻ�������** ��ͨ������ɭ������Thompson Sampling������������ȫ�ַ��������Դ������Ǳ���ľֲ�����


## �������

### ����ļ�˵��

�Ż���������� `results/{function_name}/{timestamp}/` Ŀ¼�£�

- `best_result.txt`: ���Ž������ֵ
- `convergence_plot.png`: ��������ͼ
- `gp_predictions.png`: GP Ԥ�����ܷ���
- `tr_performance.png`: �����������ܶԱ�
- `exploration_plot.png`: ����·�����ӻ�
- `iteration_data.pkl`: ��ϸ�ĵ�������


## ԭʼ��������

����Ŀ������������ʵ�֣�

**Scalable Global Optimization via Local Bayesian Optimization**
*David Eriksson, Michael Pearce, Jacob Gardner, Ryan D Turner, Matthias Poloczek*
*Advances in Neural Information Processing Systems (NeurIPS), 2019*

��������: http://papers.nips.cc/paper/8788-scalable-global-optimization-via-local-bayesian-optimization

```bibtex
@inproceedings{eriksson2019scalable,
  title = {Scalable Global Optimization via Local {Bayesian} Optimization},
  author = {Eriksson, David and Pearce, Michael and Gardner, Jacob and Turner, Ryan D and Poloczek, Matthias},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {5496--5507},
  year = {2019},
  url = {http://papers.nips.cc/paper/8788-scalable-global-optimization-via-local-bayesian-optimization.pdf}
}
```
