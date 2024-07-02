// Abrir y Cerra Cart

const cartIcon = document.querySelector("#cart-icon");
const cart = document.querySelector(".cart");
const closeCart = document.querySelector("#cart-close");

cartIcon.addEventListener("click", () => {
    cart.classList.add("active");
});

closeCart.addEventListener("click", () => {
    cart.classList.remove("active");
});

// Comenzar cuando el documento este listo

if(document.readyState == 'loading'){
    document.addEventListener("DOMContentLoaded", start);
} else {
    start();
}

// Comenzar
function start() {
    addEvents();
}

// Actualizar y volver a presentar
function update(){
    addEvents();
    updateTotal();
}

// Eventos
function addEvents(){
    //Quitar articulos del carrito
    let cartRemove_btns = document.querySelectorAll(".cart-remove");

    console.log(cartRemove_btns);

    cartRemove_btns.forEach((btn) => {
        btn.addEventListener("click", handle_removeCartItem);
    });

    // Cambiar cantidad de articulos
    let cartQuantity_inputs = document.querySelectorAll(".cart-quantity");

    cartQuantity_inputs.forEach((input) => {
        input.addEventListener("change", handle_changeItemQuantity);
    });

    // Añadir articulos al carrito
    let addCart_btns = document.querySelectorAll(".add-cart");
    addCart_btns.forEach((btn) => {
        btn.addEventListener("click", handle_addCartItem);
    });
}

// Comprar orden
const buy_btn = document.querySelector(".btn-buy");
buy_btn.addEventListener("click", handle_buyOrden);

// Funcion de manejos de eventos
let itemsAdded = [];

function handle_addCartItem(){
    let product = this.parentElement;
    let title = product.querySelector(".product-title").innerHTML;
    let price = product.querySelector(".product-price").innerHTML;
    let imgSrc = product.querySelector(".product-img").src;

    console.log(title, price, imgSrc);
    
    let newToAdd = {
    title,
    price,
    imgSrc,
    };

    /**/
    //El elemento de manejo ya existe
    if(itemsAdded.find((el) => el.title == newToAdd.title)){
        alert("Este Articulo ya existe");
        return;
    }else{
        itemsAdded.push(newToAdd);
    }

    // Añadir productos al carrito

    let carBoxElement = cartBoxComponent(title, price, imgSrc);
    let newNode = document.createElement("div");
    newNode.innerHTML = carBoxElement;
    const cartContent = cart.querySelector(".cart-content");
    cartContent.appendChild(newNode);

    update();
}

function handle_removeCartItem(){
    this.parentElement.remove();

    itemsAdded = itemsAdded.filter(
        (el) => 
            el.title != this.parentElement.querySelector(".cart-product-title").innerHTML
    );

    update();
}

function handle_changeItemQuantity(){
    if(isNaN(this.value) || this.value < 1){
        this.value = 1;
    }

    this.value = Math.floor(this.value); // Para mantener el numero entero

    update();
}

function handle_buyOrden(){
    if (itemsAdded.length <= 0) {
        alert("Aun no hay ningun pedido para realizar! \nPor favor, haga un pedido primero");
        return;
    }
    
    const cartContent = cart.querySelector(".cart-content");
    cartContent.innerHTML = "";
    alert("Su pedido se realizo con exito :)");
    itemsAdded = [];

    update();
}

// Funciones para actualizar y  renderizar

function updateTotal() {
    let cartBoxes = document.querySelectorAll(".cart-box");
    const totalElement = cart.querySelector(".total-price");
    let total = 0;

    cartBoxes.forEach((cartBox) => {
        let priceElement = cartBox.querySelector(".cart-price");
        let price = parseFloat(priceElement.innerHTML.replace("$", ""));
        let quantity = cartBox.querySelector(".cart-quantity").value;

        total += price * quantity;
    });

    total = total.toFixed(2);

    // Mantener 2 dijitos despues del punto decimal

    totalElement.innerHTML = "$" + total;
}

// Componentes HTML

function cartBoxComponent(title, price, imgSrc) {
    return `
    <div class="cart-box">
        <img src="${imgSrc}" alt="" class="cart-img">

        <div class="detail-box"> 
            <div class="cart-product-title">${title}</div>
            <div class="cart-price">${price}</div>
            <input type="number" value="1" class="cart-quantity">
        </div>

        <!-- Eliminar cart -->
        <i class='bx bxs-trash-alt cart-remove'></i>
    </div>
    `;
}
