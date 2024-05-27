
list_into_gpt = ['',
 '    H2: Phối màu trong thiết kế là gì và tại sao nó quan trọng?',
 '',
 '    H2: 6 quy tắc phối màu cơ bản trong thiết kế',
 '',
 '    H3: Phối màu đơn sắc (Monochromatic): Tạo ra sự thống nhất và đơn giản',
 '    H3: Phối màu tương đồng (Analogous): Tạo ra sự hài hoà và dễ chịu',
 '    H3: Phối màu tương phản (Complementary): Tạo ra sự nổi bật và tương phản',
 '    H3: Phối màu bổ túc bộ ba (Triadic): Tạo ra sự cân bằng và đa dạng',
 '    H3: Phối màu bổ túc xen kẽ (Split – complementary): Tạo ra sự tinh tế và hấp dẫn',
 '    H3: Phối màu bổ túc bộ bốn (Rectangular Tetradic/Compound Complementary): Tạo ra sự độc đáo và sáng tạo',
 '',
 '',
 '    H2: Cách áp dụng nguyên tắc phối màu theo từng lĩnh vực thiết kế',
 '    H3: Phối màu trong thiết kế nội thất: Tạo không gian sống đẹp và thoải mái',
 '    H3: Phối màu trong thiết kế thời trang: Tạo phong cách thời trang cá tính và ấn tượng',
 '    H3: Phối màu trong thiết kế web: Tạo website chuyên nghiệp và thu hút người dùng',
 '    H3: Phối màu trong thiết kế logo: Tạo logo độc đáo và gây ấn tượng',
 '',
 '    H2: 5 Mẹo phối màu trong thiết kế tạo nên sự thu hút',
 '',
 '    H3: Sử dụng bảng màu',
 '    H3: Thử nghiệm với các màu sắc khác nhau',
 '    H3: Chú ý đến ánh sáng',
 '    H3: Cân bằng giữa màu sắc',
 '    H3: Tạo điểm nhấn',
 ''] 


list_clear = []

list_into_gpt = [x for x in list_into_gpt if str(x) != '']

i = 0
while i in range(len(list_into_gpt)):
    text_pro = list_into_gpt[i]
    if 'H2' in list_into_gpt[i]:
        if len(list_into_gpt) - i <3:
            count = 0
            while count < len(list_into_gpt) - i:
                count = count + 1
                try:
                    if 'H2' in list_into_gpt[i + count]:
                        list_clear.append(text_pro)
                        i = i + count
                        break
                    elif i+count == len(list_into_gpt):
                        text_pro = text_pro + '\n' + list_into_gpt[i + count]
                        list_clear.append(text_pro)
                    else:
                        text_pro = text_pro + '\n' + list_into_gpt[i + count]
                except:
                    list_clear.append(text_pro)                
            i = i + count + 1
        else:       
            count = 0
            while count <3:
                count = count + 1
                if 'H2' in list_into_gpt[i + count]:
                    list_clear.append(text_pro)
                    i = i + count
                    break
                else:
                    text_pro = text_pro + '\n' + list_into_gpt[i + count]
                    if count == 3:
                        list_clear.append(text_pro)
                        i = i + count + 1
    elif 'H3' in text_pro:
        if len(list_into_gpt) - i <3:
            count = 0
            while count < len(list_into_gpt) - i:
                count = count + 1
                try:
                    if 'H2' in list_into_gpt[i + count]:
                        list_clear.append(text_pro)
                        i = i + count
                        break
                    elif i+count == len(list_into_gpt):
                        text_pro = text_pro + '\n' + list_into_gpt[i + count]
                        list_clear.append(text_pro)
                    else:
                        text_pro = text_pro + '\n' + list_into_gpt[i + count]
                except:
                    list_clear.append(text_pro)
                
            i = i + count + 1
        else:
            count = 0 
            while count <3:
                count = count + 1
                if 'H2' in list_into_gpt[i + count]:
                    list_clear.append(text_pro)
                    i = i + count
                    break
                else:
                    text_pro = text_pro + '\n' + list_into_gpt[i + count]
                    if count == 3:
                        list_clear.append(text_pro)
                        i = i + count + 1
                    