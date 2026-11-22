import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const here = path.dirname(fileURLToPath(import.meta.url));
const payload = JSON.parse(await fs.readFile(path.join(here,'table_payload.json'),'utf8'));
const output = payload.output;
await fs.mkdir(output,{recursive:true});
const workbook = Workbook.create();
const metadata = [];
function col(index) {
  let value='';
  for(let n=index;n>0;n=Math.floor((n-1)/26))value=String.fromCharCode(65+(n-1)%26)+value;
  return value;
}

for(const dataset of payload.datasets) {
  const sheet=workbook.worksheets.add(dataset.name);
  sheet.showGridLines=false;
  const lastCol=col(2+4*payload.ratios.length);
  const lastRow=8+dataset.data_rows+dataset.modes.length;
  const whole=sheet.getRange(`A1:${lastCol}${lastRow}`);
  whole.format.font={name:'Microsoft YaHei',size:10,color:'#111111'};
  whole.format.fill='#FFFFFF';
  whole.format.rowHeightPx=25;
  whole.format.verticalAlignment='center';
  whole.format.wrapText=false;
  sheet.getRange(`A1:A${lastRow}`).format.columnWidthPx=270;
  sheet.getRange(`B1:B${lastRow}`).format.columnWidthPx=185;
  sheet.getRange(`C1:${lastCol}${lastRow}`).format.horizontalAlignment='center';
  sheet.getRange('A1').format.rowHeightPx=12;
  sheet.mergeCells(`A2:${lastCol}2`);
  sheet.getRange('A2').values=[[`${dataset.name} 各场景渲染质量与Gaussian数量`]];
  sheet.getRange(`A2:${lastCol}4`).format.horizontalAlignment='left';
  sheet.getRange('A2').format.font={size:15,bold:true};
  sheet.getRange('A2').format.rowHeightPx=35;
  sheet.mergeCells(`A3:${lastCol}3`);
  sheet.getRange('A3').values=[['#GS：局部取全部评价视角选点ID的并集，重复只计一次；全局和删除方法取固定保留数。']];
  sheet.mergeCells(`A4:${lastCol}4`);
  sheet.getRange('A4').values=[['质量沿用未校正原图的既有全轨迹评价。PSNR单位dB；SSIM/LPIPS无量纲；#GS单位为点。']];
  sheet.getRange(`A3:${lastCol}4`).format.font={size:10,color:'#444444'};
  sheet.mergeCells('A5:A6');sheet.getRange('A5').values=[['场景']];
  sheet.mergeCells('B5:B6');sheet.getRange('B5').values=[['方法']];
  const header=sheet.getRange(`A5:${lastCol}6`);
  header.format.font={size:10,bold:true};
  header.format.horizontalAlignment='center';
  header.format.borders={top:{style:'medium',color:'#111111'},bottom:{style:'thin',color:'#111111'}};
  header.format.rowHeightPx=25;
  for(let i=0;i<payload.ratios.length;i++) {
    const start=3+i*4,end=start+3;
    const left=col(start),right=col(end);
    sheet.mergeCells(`${left}5:${right}5`);
    sheet.getRange(`${left}5`).values=[[payload.ratios[i]]];
    sheet.getRange(`${left}5`).setNumberFormat('0%');
    sheet.getRange(`${left}6:${right}6`).values=[['PSNR ↑','SSIM ↑','LPIPS ↓','#GS']];
    for(let j=0;j<4;j++) {
      const column=col(start+j),range=sheet.getRange(`${column}7:${column}${lastRow}`);
      range.setNumberFormat(j===0?'0.00':j===3?'#,##0':'0.000');
      sheet.getRange(`${column}1:${column}${lastRow}`).format.columnWidthPx=[70,72,75,87][j];
    }
    sheet.getRange(`${left}5:${left}${lastRow}`).format.borders={left:{style:'thin',color:'#A0A0A0'}};
  }
  let current=7;
  const dataRows=[];const sections=[];
  for(const mode of dataset.modes) {
    const groups=dataset.groups.filter(g=>g.mode===mode);
    sheet.mergeCells(`A${current}:${lastCol}${current}`);
    sheet.getRange(`A${current}`).values=[[groups[0].mode_label]];
    const section=sheet.getRange(`A${current}:${lastCol}${current}`);
    section.format.fill='#F1F1F1';section.format.font={bold:true};
    section.format.horizontalAlignment='left';
    section.format.borders={top:{style:'thin',color:'#111111'},bottom:{style:'thin',color:'#B0B0B0'}};
    section.format.rowHeightPx=29;
    sections.push({mode,row:current});
    current++;
    for(const group of groups) {
      const first=current;
      const matrix=group.methods.map(method=>[
        group.scene,method.label,...method.values.flatMap(v=>[v.psnr,v.ssim,v.lpips,v.gs])
      ]);
      sheet.getRange(`A${first}:${lastCol}${first+3}`).values=matrix;
      sheet.mergeCells(`A${first}:A${first+3}`);
      sheet.getRange(`A${first}`).format.font={bold:true};
      for(let i=0;i<4;i++) {
        const method=group.methods[i];
        if(method.method.startsWith('ours_'))sheet.getRange(`B${first+i}`).format.font={bold:true};
        dataRows.push({row:first+i,scene:group.scene,mode,method:method.method});
      }
      sheet.getRange(`A${first+3}:${lastCol}${first+3}`).format.borders={bottom:{style:'thin',color:'#B0B0B0'}};
      current+=4;
    }
  }
  sheet.getRange(`A${current-1}:${lastCol}${current-1}`).format.borders={bottom:{style:'medium',color:'#111111'}};
  sheet.freezePanes.freezeRows(6);
  sheet.freezePanes.freezeColumns(2);
  metadata.push({name:dataset.name,tag:dataset.tag,rows:current-1,columns:46,dataRows,sections});
  console.log('SHEET_POPULATED',dataset.name,dataset.data_rows,'rows');
}

workbook.recalculate();
for(const meta of metadata) {
  const check=await workbook.inspect({kind:'region',sheetId:meta.name,range:'A5:J11',
    include:'values,formulas',maxChars:1800,tableMaxRows:7,tableMaxCols:10});
  await fs.writeFile(path.join(here,`${meta.tag}_inspect.ndjson`),check.ndjson,'utf8');
  const sample=await workbook.render({sheetName:meta.name,range:'A5:R11',scale:2,format:'png',headers:false});
  await fs.writeFile(path.join(here,`${meta.tag}_sample.png`),new Uint8Array(await sample.arrayBuffer()));
}
const errors=await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options:{useRegex:true,maxResults:50},summary:'表格错误扫描',maxChars:2000});
await fs.writeFile(path.join(here,'error_scan.ndjson'),errors.ndjson,'utf8');
const xlsx=await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(output,'三数据集_各场景Ratio质量与去重点数.xlsx'));
await fs.rename(path.join(output,'三数据集_各场景Ratio质量与去重点数.xlsx.inspect.ndjson'),
                path.join(here,'export_inspect.ndjson'));
await fs.writeFile(path.join(here,'WORKBOOK_LAYOUT.json'),JSON.stringify(metadata,null,2)+'\n','utf8');
for(const meta of metadata) {
  console.log('RENDERING',meta.tag,`A2:AT${meta.rows}`);
  const preview=await workbook.render({sheetName:meta.name,range:`A2:AT${meta.rows}`,scale:2,format:'png',headers:false});
  await fs.writeFile(path.join(output,`${meta.tag}_全部场景表.png`),new Uint8Array(await preview.arrayBuffer()));
}
console.log('TABLES_EXPORTED',metadata.length,'sheets');
