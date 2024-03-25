console.log(123)
// 查找footer元素
var footerElement = document.querySelector("footer");
console.log("试图删除footer");
// 如果找到了footer元素，则删除它
if (footerElement) {
    footerElement.parentNode.removeChild(footerElement);
} else {
    console.log("未找到footer元素。");
}