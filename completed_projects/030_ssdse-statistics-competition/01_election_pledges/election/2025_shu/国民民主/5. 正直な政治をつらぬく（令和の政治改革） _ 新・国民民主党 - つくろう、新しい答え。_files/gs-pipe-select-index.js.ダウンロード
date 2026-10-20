document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll('select').forEach(function (select) {
        // Create a hidden input field
        let hiddenInput = document.createElement("input");
        hiddenInput.type = "hidden";
        hiddenInput.name = select.name + "-gsc-pipe-selected-index";
        hiddenInput.value = select.selectedIndex;

        // Append the hidden input next to the select
        select.parentNode.appendChild(hiddenInput);

        // Update hidden input on select change
        select.addEventListener("change", function () {
            hiddenInput.value = select.selectedIndex;
        });
    });
});
