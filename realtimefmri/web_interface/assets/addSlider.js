
var sliderDiv = document.createElement("div");
sliderDiv.id = "slider-div";
var slider = document.createElement("input");
slider.type = "range";
slider.id = "graph-height-slider";
slider.style.width = "90vw";
slider.min = 400; slider.max = 1600; slider.value = 400;
sliderDiv.append(slider);
document.body.prepend(sliderDiv);

slider.oninput = function () {
	var c = document.getElementById("graph-div");
	c.style.height = this.value + "px";
}