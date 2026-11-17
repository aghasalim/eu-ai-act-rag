// The page is complete without this file: it only shows the result panels
// that match the ticked boxes and keeps the selection in the URL.
const boxes = [...document.querySelectorAll('#form input[type=checkbox]')];
const ticked = g => boxes.filter(b => b.checked && b.dataset.group === g);

function update() {
  const any = g => ticked(g).length > 0;
  const exempt = any('annex3') && any('derogation') && !any('profiling') && !any('annex1');
  const show = {
    prohibited: any('prohibited'),
    highrisk: (any('annex1') || any('annex3')) && !exempt,
    derogation: exempt,
    transparency: any('transparency'),
    gpai: any('gpai'),
  };
  show.none = !Object.values(show).some(Boolean);
  document.querySelectorAll('[data-show]').forEach(el => {
    el.hidden = !show[el.dataset.show];
  });
  document.querySelectorAll('[data-list]').forEach(el => {
    el.innerHTML = el.dataset.list.split(' ').flatMap(ticked)
      .map(b => `<a href="${b.dataset.href}">${b.dataset.ref}</a>`).join(', ');
  });
  document.getElementById('tags').innerHTML =
    [...document.querySelectorAll('.verdict:not([hidden])')].map(v =>
      `<span class="tag ${v.classList[1]}">${v.querySelector('h3').textContent}</span>`).join(' ');
  const state = boxes.filter(b => b.checked).map(b => b.value).join(',');
  history.replaceState(null, '', state ? '#' + state : location.pathname);
}

const saved = new Set(decodeURIComponent(location.hash.slice(1)).split(','));
boxes.forEach(b => { b.checked = saved.has(b.value); b.addEventListener('change', update); });
document.getElementById('form').addEventListener('reset', () => setTimeout(update));
document.getElementById('copy').addEventListener('click', async e => {
  await navigator.clipboard.writeText(location.href);
  e.target.textContent = 'Link copied';
});
update();
