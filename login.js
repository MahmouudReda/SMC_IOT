function setFormMessage(formElement, type, message){
    const messageElement=formElement.querySelector(".form__message");
    messageElement.textContent=message;
    messageElement.classList.remove("form__message--success","form__message--error");
    messageElement.classList.add(`form__message--${type}`);
    
}
function endsWith(str, suffix) {
    return str.indexOf(suffix, str.length - suffix.length) !== -1;
}

function setInputError(inputElement,message){
    inputElement.classList.add("form__input--error");
    inputElement.parentElement.querySelector(".form__input-error-message").textContent=message;

}
function clearInputError(inputElement){
    inputElement.classList.remove("form__input--error");
    inputElement.parentElement.querySelector(".form__input-error-message").textContent="";

}
document.addEventListener("DOMContentLoaded",()=>{
    
    const loginForm= document.querySelector("#login");
    const CreateAccountForm= document.querySelector("#CreateAcount");

    document.querySelector("#LinkCreatAccount").addEventListener("click",e=>{
        e.preventDefault();
        loginForm.classList.add("form__hidden");
        CreateAccountForm.classList.remove("form__hidden");
    });

    document.querySelector("#LinkLogin").addEventListener("click",e=>{
        e.preventDefault();
        loginForm.classList.remove("form__hidden");
        CreateAccountForm.classList.add("form__hidden");
    });
    /*loginForm.addEventListener("submit",e=>{
        e.preventDefault();
        setFormMessage(loginForm,"error","Invalid Email or password");
    });*/
    document.querySelectorAll(".form__input").forEach(inputElement => {
        inputElement.addEventListener("blur",e=>{
            if(e.target.id==="Username" && e.target.value.length>0 && e.target.value.length<10){
                setInputError(inputElement, "username must be more than 10 character");
            }
        });

    inputElement.addEventListener("input", e=>{
        clearInputError(inputElement);
    }); 
    });

    document.querySelectorAll(".form__input").forEach(inputElement => {
        inputElement.addEventListener("blur",e=>{
            if(e.target.id==="password" && e.target.value.length>0 && e.target.value.length<8){
                setInputError(inputElement, "password must be more than 8 character");
            }
        });
        

    inputElement.addEventListener("input", e=>{
        clearInputError(inputElement);
    }); 
    });

    document.querySelectorAll(".form__input").forEach(inputElement => {
        inputElement.addEventListener("blur",e=>{
            const text=document.getElementById("email");
            if(e.target.id==="email" && !(endsWith(e.target.value,"@fci.bu.edu.eg"))){
                setInputError(inputElement, "email must belong to BFCAI");
            }

        });

    inputElement.addEventListener("input", e=>{
        clearInputError(inputElement);
    }); 
    });
    

    document.querySelectorAll(".form__input").forEach(inputElement => {
       
        inputElement.addEventListener("blur",e=>{

            if(e.target.id === "repassword" && document.getElementById("password").value !== e.target.value){
                setInputError(inputElement, `password not match`);
            }
        });

    inputElement.addEventListener("input", e=>{
        clearInputError(inputElement);
    }); 
    });


});

/*
 if(e.target.id==="password" && e.target.value.length>0 && e.target.value.length<8){
                setInputError(inputElement, "password must be more than 8 character");
            }
if(e.target.id==="password" && !e.target.id==="repassword"){
                setInputError(inputElement, "username must be more than 10 character");
            }
*/