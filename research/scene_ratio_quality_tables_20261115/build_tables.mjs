import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';

const here=path.dirname(fileURLToPath(import.meta.url));
const payload=JSON.parse(await fs.readFile(path.join(here,'table_payload.json'),'utf8'));
const backup=path.join(here,'before_full_20261115');
const filename='三数据集_各场景Ratio质量与去重点数.xlsx';
const workbook=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(backup,'delivery',filename)));
const layout=JSON.parse(await fs.readFile(path.join(backup,'WORKBOOK_LAYOUT.json'),'utf8'));

for(const meta of layout) {
  const sheet=workbook.worksheets.getItem(meta.name);
  const dataset=payload.datasets.find(d=>d.tag===meta.tag);
  const added=sheet.getRange(`AU5:AX${meta.rows}`);
  added.format.font={name:'Microsoft YaHei',size:10,color:'#111111'};
  added.format.fill='#FFFFFF';
  added.format.verticalAlignment='center';
  added.format.horizontalAlignment='center';
  added.format.wrapText=false;
  for(const [column,width,format] of [['AU',70,'0.00'],['AV',72,'0.000'],['AW',75,'0.000'],['AX',87,'#,##0']]) {
    sheet.getRange(`${column}1:${column}${meta.rows}`).format.columnWidthPx=width;
    sheet.getRange(`${column}7:${column}${meta.rows}`).setNumberFormat(format);
  }
  for(const row of [2,3,4]) {
    const value=sheet.getRange(`A${row}`).values[0][0];
    sheet.unmergeCells(`A${row}:AT${row}`);
    sheet.mergeCells(`A${row}:AX${row}`);
    sheet.getRange(`A${row}`).values=[[value]];
    sheet.getRange(`A${row}:AX${row}`).format.horizontalAlignment='left';
  }
  sheet.getRange('A3').values=[['#GS：局部取全部评价视角选点ID的并集；全局和删除方法取固定保留数；Full取完整地图N。']];
  sheet.getRange('A4').values=[['Full取未选择、未补偿的完整地图：两种CaRtGS方法共用原版Full，两种我们方法共用当前模型Full。质量沿用原图评价。']];
  sheet.mergeCells('AU5:AX5');
  sheet.getRange('AU5').values=[['Full']];
  sheet.getRange('AU6:AX6').values=[['PSNR ↑','SSIM ↑','LPIPS ↓','#GS']];
  sheet.getRange('AU5:AX6').format.font={bold:true};
  sheet.getRange('AU5:AX6').format.borders={top:{style:'medium',color:'#111111'},bottom:{style:'thin',color:'#111111'}};
  sheet.getRange(`AU5:AU${meta.rows}`).format.borders={left:{style:'thin',color:'#A0A0A0'}};
  for(const section of meta.sections) {
    const row=section.row;
    const label=sheet.getRange(`A${row}`).values[0][0];
    sheet.unmergeCells(`A${row}:AT${row}`);
    sheet.mergeCells(`A${row}:AX${row}`);
    sheet.getRange(`A${row}`).values=[[label]];
    const range=sheet.getRange(`A${row}:AX${row}`);
    range.format.fill='#F1F1F1';range.format.font={bold:true};
    range.format.horizontalAlignment='left';
    range.format.borders={top:{style:'thin',color:'#111111'},bottom:{style:'thin',color:'#B0B0B0'}};
  }
  for(const row of meta.dataRows) {
    const item=dataset.groups.find(g=>g.scene===row.scene&&g.mode===row.mode).methods.find(m=>m.method===row.method);
    const full=item.values.at(-1);
    if(full.ratio!==1)throw new Error('Full value missing');
    sheet.getRange(`AU${row.row}:AX${row.row}`).values=[[full.psnr,full.ssim,full.lpips,full.gs]];
    if(row.method==='ours_view_ratio')sheet.getRange(`AU${row.row}:AX${row.row}`).format.borders={bottom:{style:'thin',color:'#B0B0B0'}};
  }
  sheet.getRange(`AU${meta.rows}:AX${meta.rows}`).format.borders={bottom:{style:'medium',color:'#111111'}};
  meta.columns=50;
  console.log('FULL_ADDED',meta.name,meta.dataRows.length,'rows');
}

workbook.recalculate();
for(const meta of layout) {
  const check=await workbook.inspect({kind:'region',sheetId:meta.name,range:'AQ5:AX11',
    include:'values,formulas',maxChars:1800,tableMaxRows:7,tableMaxCols:8});
  await fs.writeFile(path.join(here,`${meta.tag}_full_inspect.ndjson`),check.ndjson,'utf8');
  const sample=await workbook.render({sheetName:meta.name,range:'AQ5:AX11',scale:2,format:'png',headers:false});
  await fs.writeFile(path.join(here,`${meta.tag}_full_sample.png`),new Uint8Array(await sample.arrayBuffer()));
}
const errors=await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options:{useRegex:true,maxResults:50},summary:'表格错误扫描',maxChars:2000});
await fs.writeFile(path.join(here,'error_scan.ndjson'),errors.ndjson,'utf8');
const xlsx=await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(payload.output,filename));
await fs.rename(path.join(payload.output,filename+'.inspect.ndjson'),path.join(here,'export_inspect.ndjson'));
await fs.writeFile(path.join(here,'WORKBOOK_LAYOUT.json'),JSON.stringify(layout,null,2)+'\n','utf8');
for(const meta of layout) {
  const preview=await workbook.render({sheetName:meta.name,range:`A2:AX${meta.rows}`,scale:2,format:'png',headers:false});
  await fs.writeFile(path.join(payload.output,`${meta.tag}_全部场景表.png`),new Uint8Array(await preview.arrayBuffer()));
}
console.log('FULL_TABLES_EXPORTED',layout.length,'sheets');
