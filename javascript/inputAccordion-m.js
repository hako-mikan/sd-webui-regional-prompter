function setupAccordion_m(accordion) {
    // 既に処理済みの場合はスキップ
    if (accordion.getAttribute('data-processed') === 'true') {
        return;
    }
    accordion.setAttribute('data-processed', 'true'); // 処理済みフラグを追加

    var labelWrap = accordion.querySelector('.label-wrap');
    var gradioCheckbox = gradioApp().querySelector('#' + accordion.id + "-checkbox input");
    var extra = gradioApp().querySelector('#' + accordion.id + "-extra");
    var span = labelWrap.querySelector('span');

    // 初期状態をGradioの状態に基づいて設定
    var visibleCheckbox = document.createElement('INPUT');
    visibleCheckbox.type = 'checkbox';
    visibleCheckbox.checked = gradioCheckbox.checked; // Gradioの初期状態を継承
    visibleCheckbox.id = accordion.id + "-visible-checkbox";
    visibleCheckbox.className = gradioCheckbox.className + " input-accordion-checkbox";

    // 既にチェックボックスが存在していないか確認して追加
    if (!span.querySelector(`#${visibleCheckbox.id}`)) {
        span.insertBefore(visibleCheckbox, span.firstChild);
    }

    accordion.visibleCheckbox = visibleCheckbox;

    if (extra) {
        labelWrap.insertBefore(extra, labelWrap.lastElementChild);
    }

    // チェックボックスクリック時のイベント
    visibleCheckbox.addEventListener('click', function(event) {
        event.stopPropagation(); // クリックイベントの伝播を停止
        console.log(`Checkbox in accordion ${accordion.id} is now: `, visibleCheckbox.checked);

        // Gradioのチェックボックス状態を更新
        gradioCheckbox.checked = visibleCheckbox.checked;
        gradioCheckbox.dispatchEvent(new Event('input', { bubbles: true }));
    });

    // Gradioチェックボックスの変更を監視して表示用チェックボックスを更新
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            visibleCheckbox.checked = gradioCheckbox.checked;
            console.log(`Visible checkbox in accordion ${accordion.id} updated to: `, visibleCheckbox.checked);
        });
    });

    observer.observe(gradioCheckbox, { attributes: true, attributeFilter: ['checked'] });
}

onUiLoaded(function() {
    for (var accordion of gradioApp().querySelectorAll('.input-accordion-m')) {
        setupAccordion_m(accordion);
    }
});
