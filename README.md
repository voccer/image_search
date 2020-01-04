## Giới thiệu
Ứng dụng tìm kiếm hình ảnh quần áo

## Hướng dẫn sử dụng
<ul>
    <li>Tạo thư mục model/ranknet_mix1</li>
    <li>Tải 2 file annoy và weights từ:</li>
    <ul>
        <li>https://drive.google.com/open?id=1hLzDgvrmgx3xpC_X_6CcNOkNZFmVtwzk</li>
        <li>https://drive.google.com/open?id=1Rupz4g0AVqXCOYgMM-2j_fGoyxhq_hc5</li>
    </ul>
    <li> Cho 2 file vừa tải vào ranknet_mix1 </li>
    <li>Tải dataset từ link: </li>
    <ul>
        <li> https://drive.google.com/open?id=1ADQI52KIEI6gE5-SUeZKzlFbrXUCPrwt </li>
    </ul>
    <li>Giải nén vào thư mục static/dataset/images</li>
    <li> <code>pip install -r requirements.txt</code> </li>
    <li><code>flask run</code></li>(đảm bảo port: 5000 vẫn còn khả dụng)
    <li>Vào browser và vào đường link <code>localhost:5000</code> </li>
</ul>

## Công nghệ
<strong>Ứng dụng sử dụng 2 version:</strong>
<ul>
    <li>Sử dụng template jinja2 của python - <code>git checkout jinja2</code>  </li>
    <li>Sử dụng ajax xử lý không load lại trang - <code>git checkout ajax</code> </li>
</ul>