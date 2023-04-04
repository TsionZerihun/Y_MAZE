

const toggleBtnSap = document.querySelector('#toggleBtnSap');
const divListSap = document.querySelector('#sapList');

// action to be taken when clicked on hide SAP button
toggleBtnSap.addEventListener('click', () => {
  if (divListSap.style.display === 'none') {
    divListSap.style.display = 'block';
    toggleBtnSap.innerHTML = 'Hide SAP';
  } else {
    divListSap.style.display = 'none';
    toggleBtnSap.innerHTML = 'SAP';
  }
});


const toggleBtnSar = document.querySelector('#toggleBtnSar');
const divListSar = document.querySelector('#sarList');

// action to be taken when clicked on hide SAR button
toggleBtnSar.addEventListener('click', () => {
  if (divListSar.style.display === 'none') {
    divListSar.style.display = 'block';
    toggleBtnSar.innerHTML = 'Hide SAR';
  } else {
    divListSar.style.display = 'none';
    toggleBtnSar.innerHTML = 'SAR';
  }
});

const toggleBtnAar = document.querySelector('#toggleBtnAar');
const divListAar = document.querySelector('#aarList');

// action to be taken when clicked on hide SAR button
toggleBtnAar.addEventListener('click', () => {
  if (divListAar.style.display === 'none') {
    divListAar.style.display = 'block';
    toggleBtnAar.innerHTML = 'Hide AAR';
  } else {
    divListAar.style.display = 'none';
    toggleBtnAar.innerHTML = 'AAR';
  }
});
