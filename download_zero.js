
var data = source.data;
var filetext = 'ID,Class,Territory,Date,Act,Pred,Low,High,Trend,Slope,Change,Error,IDX\n';
for (i=0; i < data['ID'].length; i++) {
    var currRow = [data['ID'][i].toString(),
                   data['Class'][i].toString(),
			    data['Territory'][i].toString(),
			    data['Date'][i].toString(),
			    data['Act'][i].toString(),
			    data['Pred'][i].toString(),
			    data['Low'][i].toString(),
			    data['High'][i].toString(),
			    data['Trend'][i].toString(),
			    data['Slope'][i].toString(),
                   data['Change'][i].toString(),
			    data['Error'][i].toString(),
			    data['IDX'][i].toString()
			    .concat('\n')];

    var joined = currRow.join();
    filetext = filetext.concat(joined);
}

var filename = 'Class.csv';
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
}

else {
    var link = document.createElement("a");
    link = document.createElement('a')
    link.href = URL.createObjectURL(blob);
    link.download = filename
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}
