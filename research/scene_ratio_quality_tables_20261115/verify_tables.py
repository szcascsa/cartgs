"""独立读取最终Excel、CSV并核查原始数值、去重点数与覆盖，不改写Excel。"""
from pathlib import Path
from copy import copy
import csv
import hashlib
import json
import zipfile

import openpyxl
from PIL import Image

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))


def main():
    payload = json.loads((ROOT / 'table_payload.json').read_text(encoding='utf-8'))
    layout = json.loads((ROOT / 'WORKBOOK_LAYOUT.json').read_text(encoding='utf-8'))
    output = Path(payload['output'])
    book_path = output / '三数据集_各场景Ratio质量与去重点数.xlsx'
    book = openpyxl.load_workbook(book_path, data_only=False)
    previous = openpyxl.load_workbook(ROOT / 'before_full_20261115/delivery' / book_path.name, data_only=False)
    protected = json.loads((ROOT / 'before_full_20261115/FILES_BEFORE.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT / 'before_full_20261115/delivery' / name) == value for name,value in protected.items())
    assert book.sheetnames == ['Replica', 'TUM_RGBD', 'VECtor']
    assert all(sha(Path(path)) == expected for path, expected in payload['sources'].items())
    checked = counts = entries = prior_cells = 0
    manifest = []
    for dataset, meta in zip(payload['datasets'], layout):
        sheet = book[dataset['name']]
        prior = previous[dataset['name']]
        assert sheet.freeze_panes == 'C7' and sheet.sheet_view.showGridLines is False
        for column in range(1,47):
            from openpyxl.utils import get_column_letter
            letter = get_column_letter(column)
            assert sheet.column_dimensions[letter].width == prior.column_dimensions[letter].width
        for row in range(1,meta['rows']+1):
            assert sheet.row_dimensions[row].height == prior.row_dimensions[row].height
            for column in range(1,47):
                before,after = prior.cell(row,column),sheet.cell(row,column)
                if row not in [3,4]:
                    assert before.value == after.value,(dataset['name'],after.coordinate,'旧内容改变')
                if row >= 5 and row not in [s['row'] for s in meta['sections']]:
                    assert copy(before.font) == copy(after.font) and copy(before.fill) == copy(after.fill)
                    assert copy(before.alignment) == copy(after.alignment) and before.number_format == after.number_format
                prior_cells += 1
        lookup = {(g['scene'], g['mode'], item['method']): item
                  for g in dataset['groups'] for item in g['methods']}
        wide = rows(output / f"{dataset['tag']}_全部场景表.csv")
        flat = rows(output / f"data/{dataset['tag']}_逐项数据.csv")
        assert len(wide) == len(meta['dataRows']) == dataset['data_rows']
        assert len(flat) == dataset['data_rows'] * 12
        flat_lookup = {(r['scene'], r['mode'], r['method'], float(r['ratio'])): r for r in flat}
        assert len(flat_lookup) == len(flat)
        for record, wide_record in zip(meta['dataRows'], wide):
            row = record['row']
            item = lookup[record['scene'], record['mode'], record['method']]
            assert sheet.cell(row, 2).value == item['label'] == wide_record['方法']
            assert record['scene'] == wide_record['场景']
            for index, value in enumerate(item['values']):
                prefix = 'Full' if value['ratio'] == 1 else f"{100*value['ratio']:g}%"
                original = flat_lookup[record['scene'], record['mode'], record['method'], value['ratio']]
                for offset, (field, suffix) in enumerate([('psnr', ' PSNR(dB)'), ('ssim', ' SSIM'),
                                                         ('lpips', ' LPIPS'), ('gs', ' #GS')]):
                    cell = sheet.cell(row, 3 + 4 * index + offset)
                    assert cell.data_type == 'n' and cell.value is not None
                    assert float(cell.value).hex() == float(value[field]).hex()
                    assert float(wide_record[prefix + suffix]).hex() == float(value[field]).hex()
                    source_value = original['unique_total'] if field == 'gs' else original[field]
                    assert float(source_value).hex() == float(value[field]).hex()
                    if field == 'gs':
                        assert int(cell.value) == cell.value and cell.number_format == '#,##0'
                        counts += 1
                    else:
                        assert cell.number_format == ('0.00' if field == 'psnr' else '0.000')
                    checked += 1
                entries += 1
        for index, ratio in enumerate(payload['ratios']):
            assert sheet.cell(5, 3 + index * 4).value == ratio
        assert sheet['AU5'].value == 'Full'
        for section in meta['sections']:
            assert sheet.cell(section['row'], 1).value
        assert not any(cell.data_type == 'e' for row in sheet for cell in row)
        image_path = output / f"{dataset['tag']}_全部场景表.png"
        with Image.open(image_path) as image:
            size = image.size
            image.verify()
        manifest.append(dict(dataset=dataset['name'], scenes=len(dataset['scenes']),
                             modes=dataset['modes'], methods=4, ratios=11, full_column=True,
                             rows=len(meta['dataRows']), results=len(flat), png_size=size))
    assert (checked, counts, entries) == (6144, 1536, 1536)
    receipt = dict(complete=True, datasets=manifest, scene_mode_groups=32,
                   method_ratio_records=entries, numeric_cells_bitwise_preserved=checked,
                   original_cells_checked=prior_cells, original_ratio_values_unchanged=True,
                   full_source_records=64,full_display_rows=128,
                   full_count_definition='完整地图N；CaRtGS两行共用原版Full，我们两行共用当前模型Full',
                   integer_counts_verified=counts, all_sources_unchanged=True,
                   no_training=True, no_new_rendering=True,
                   selected_methods=['CaRtGS随机删除', 'CaRtGS不透明度删除', '我们全局Ratio', '我们局部Ratio'],
                   local_count_definition='每场景／相机模式／Ratio，对所有评价视角选中Master ID取并集，重复只计一次',
                   fixed_count_definition='全局及删除方法为固定保留集合大小',
                   sources=payload['sources'],
                   files={str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()
                          and p.name not in ['QA_RECEIPT.json', '三数据集_场景Ratio表格.zip']})
    (output / 'QA_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('TABLES_VERIFIED',entries,'records',checked,'exact numeric cells',counts,'integer counts',flush=True)


if __name__ == '__main__':
    main()
