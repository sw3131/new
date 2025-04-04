import os
import logging
import traceback
from datetime import datetime, timedelta
from flask import Flask, render_template, request, send_file, session, make_response
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config.update({
    'UPLOAD_FOLDER': 'uploads',
    'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50MB
    'TEMPLATES_AUTO_RELOAD': True
})

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class AnalysisEngine:
    """数据分析引擎"""

    REQUIRED_COLUMNS = {
        'main': ['下单日期', 'cust_id', 'bd_name', '类目', '客户名称', '商品名称', '销量'],
        'product': ['商品名称'],
        'customer': ['客户名称']
    }

    def __init__(self, params):
        self.params = params
        self.result_df = pd.DataFrame()
        self.months = []

    def validate_file(self, file_stream, file_type):
        """验证文件有效性"""
        if file_type == 'main' and not file_stream:
            raise ValueError("必须上传主数据文件")

        if not file_stream:
            return None

        # 验证文件类型
        if not file_stream.filename.lower().endswith(('.csv', '.xlsx')):
            raise ValueError("只支持 CSV 和 Excel 文件")

        # 读取文件
        try:
            if file_stream.filename.endswith('.xlsx'):
                df = pd.read_excel(file_stream)
            else:
                df = pd.read_csv(file_stream)
        except Exception as e:
            logger.error(f"文件读取失败: {str(e)}")
            raise ValueError("文件解析失败，请检查文件格式")

        # 验证必要列
        required = self.REQUIRED_COLUMNS[file_type]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{file_type}文件缺少必要列：{', '.join(missing)}")

        return df

    def process_data(self, main_file, product_file, customer_file):
        """核心数据处理流程"""

        # 读取并验证主文件
        main_df = self.validate_file(main_file, 'main')
        main_df['下单日期'] = pd.to_datetime(main_df['下单日期'])
        main_df = main_df.sort_values('下单日期')

        # 应用筛选
        if product_file:
            product_df = self.validate_file(product_file, 'product')
            product_list = product_df['商品名称'].unique()
            main_df = main_df[main_df['商品名称'].isin(product_list)]

        if customer_file:
            customer_df = self.validate_file(customer_file, 'customer')
            customer_list = customer_df['客户名称'].unique()
            main_df = main_df[main_df['客户名称'].isin(customer_list)]

        # 生成月度数据
        main_df['月份'] = main_df['下单日期'].dt.strftime('%Y年%m月')
        self.months = sorted(main_df['月份'].unique(),
                             key=lambda x: datetime.strptime(x, '%Y年%m月'))

        # 分组分析
        results = []
        grouped = main_df.groupby(['cust_id', 'bd_name', '类目', '客户名称', '商品名称'])

        for (cust_id, sales, category, customer, product), group in grouped:
            # 筛选条件
            if len(group) < self.params['min_purchases']:
                continue

            # 流失判断
            last_date = group['下单日期'].max()
            deadline = last_date + timedelta(days=self.params['churn_days'])
            if self.params['query_date'] > deadline:
                continue

            # 计算周期
            time_diff = group['下单日期'].diff().dropna()
            avg_cycle = time_diff.mean().days if not time_diff.empty else 0

            # 构建记录
            record = {
                '客户ID': cust_id,
                '业务代表': sales,
                '客户名称': customer,
                '客户匹配': '已匹配' if customer_file else '全部客户',
                '商品名称': product,
                '商品匹配': '已匹配' if product_file else '全部商品',
                '商品类目': category,
                '最后购买日期': last_date.strftime('%Y-%m-%d'),
                '最后销量': int(group[group['下单日期'] == last_date]['销量'].sum()),
                '平均周期(天)': int(avg_cycle),
                '预计购买日': (last_date + timedelta(days=avg_cycle)).strftime('%Y-%m-%d'),
                '跟进状态': '需跟进' if self.params['query_date'] >= last_date + timedelta(days=avg_cycle) else '正常'
            }

            # 添加月度销量
            monthly = group.groupby('月份')['销量'].sum()
            for month in self.months:
                record[month] = int(monthly.get(month, 0))

            results.append(record)

        # 生成DataFrame
        if results:
            base_cols = [
                '客户ID', '业务代表', '客户名称', '客户匹配',
                '商品名称', '商品匹配', '商品类目', '最后购买日期',
                '最后销量', '平均周期(天)', '预计购买日', '跟进状态'
            ]
            self.result_df = pd.DataFrame(results)[base_cols + self.months]
            self.result_df.replace({np.nan: None}, inplace=True)

        return self


@app.route('/', methods=['GET', 'POST'])
def analyze():
    try:
        if request.method == 'POST':
            # 解析参数
            params = {
                'min_purchases': max(int(request.form.get('min_purchases', 3)), 1),
                'churn_days': max(int(request.form.get('churn_days', 90)), 1),
                'query_date': datetime.strptime(request.form['query_date'], '%Y-%m-%d')
            }

            # 初始化分析引擎
            engine = AnalysisEngine(params).process_data(
                main_file=request.files.get('main_file'),
                product_file=request.files.get('product_file'),
                customer_file=request.files.get('customer_file')
            )

            # 准备结果数据
            result_df = engine.result_df
            table_data = []
            if not result_df.empty:
                table_data = result_df.fillna('').astype(str).values.tolist()

            # 生成下载文件
            download_id = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{download_id}.csv")
            result_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            session['download_id'] = download_id

            return render_template('results.html',
                                   table_html_columns=result_df.columns.tolist() if not result_df.empty else [],
                                   table_data=table_data,
                                   record_count=len(result_df))

        # GET请求显示主页
        default_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        return render_template('index.html',
                               default_date=default_date,
                               min_purchases=3,
                               churn_days=90)

    except Exception as e:
        logger.error(f"处理失败:\n{traceback.format_exc()}")
        error_msg = f"系统错误: {str(e)}" if app.debug else "处理请求时发生错误"
        return render_template('index.html',
                               error_message=error_msg,
                               prev_values=request.form)


@app.route('/download')
def download():
    try:
        if 'download_id' not in session:
            raise ValueError("下载请求已过期")

        file_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                 f"{session['download_id']}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError("文件不存在")

        response = make_response(send_file(
            file_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"销售分析报告_{datetime.now().strftime('%Y%m%d')}.csv"
        ))

        # 添加清理回调
        @response.call_on_close
        def cleanup():
            try:
                os.remove(file_path)
                session.pop('download_id', None)
            except Exception as e:
                logger.error(f"清理临时文件失败: {str(e)}")

        return response

    except Exception as e:
        logger.error(f"下载失败: {str(e)}")
        return render_template('error.html', message=str(e)), 404


@app.after_request
def add_security_headers(response):
    """添加安全头部防止缓存"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
