from pathlib import Path

from django.shortcuts import render, get_object_or_404, redirect
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Q
from django.conf import settings
from .models import Product, Category
from cart.views import _cart_id
from cart.models import CartItem
from .models import ReviewRating
from .forms import ReviewForm
from django.contrib import messages
from orders.models import OrderProduct
from .models import ProductGallery
from django.shortcuts import render
import cv2
import os
import cvzone
from cvzone.PoseModule import PoseDetector
from django.http import JsonResponse
def home(request):
    products = Product.objects.all().filter(is_available=True)
    
    context = {
        'products' : products,
    }
    return render(request, 'shop/index.html', context)


def shop(request, category_slug=None):
    categories = None
    products = None
    

    if category_slug != None:
        categories = get_object_or_404(Category, slug=category_slug)
        products = Product.objects.filter(category=categories, is_available=True)
        paginator = Paginator(products, 6)
        page = request.GET.get('page')
        paged_products = paginator.get_page(page)
        products_count = products.count()
        
    else:
        products = Product.objects.all().filter(is_available=True)
        paginator = Paginator(products, 6)
        page = request.GET.get('page')
        paged_products = paginator.get_page(page)
        products_count = products.count()
        
    
    for product in products:
        reviews = ReviewRating.objects.order_by('-updated_at').filter(product_id=product.id, status=True)

    context = {
        'category_slug': category_slug,
        'products' : paged_products,
        'products_count': products_count,
        
    }
    return render(request, 'shop/shop/shop.html', context)


def product_details(request, category_slug, product_details_slug):
    try:
        single_product = Product.objects.get(category__slug=category_slug, slug=product_details_slug)
        
        in_cart = CartItem.objects.filter(cart__cart_id=_cart_id(request), product=single_product).exists()
    except Exception as e:
        return e

    if request.user.is_authenticated:
        try:
            orderproduct = OrderProduct.objects.filter(user=request.user, product_id=single_product.id).exists()
        except OrderProduct.DoesNotExist:
            orderproduct = None
    else:
        orderproduct = None

    reviews = ReviewRating.objects.order_by('-updated_at').filter(product_id=single_product.id, status=True)
    product_gallery = ProductGallery.objects.filter(product_id=single_product.id)

    context = {
        'single_product': single_product,
        'in_cart': in_cart,
        'orderproduct':orderproduct,
        'reviews': reviews,
        'product_gallery':product_gallery,
    }
    return render(request, 'shop/shop/product_details.html', context)


def search(request):
    products_count = 0
    products = None
    paged_products = None
    if 'keyword' in request.GET:
        keyword = request.GET['keyword']
        if keyword :
            products = Product.objects.filter(Q(description__icontains=keyword) | Q(name__icontains=keyword))
            
            products_count = products.count()
            
    
    context = {
        'products': products,
        'products_count': products_count,
    }
    return render(request, 'shop/shop/search.html', context)



def review(request, product_id):
    url = request.META.get('HTTP_REFERER')
    if request.method == 'POST':
        try:
            reviews = ReviewRating.objects.get(user__id=request.user.id,product__id=product_id)
            form = ReviewForm(request.POST, instance=reviews)
            form.save()
            messages.success(request, 'Thank you, your review updated!')
            return redirect(url)
        except ReviewRating.DoesNotExist:
            form = ReviewForm(request.POST)
            if form.is_valid():
                data = ReviewRating()
                data.rating = form.cleaned_data['rating']
                data.review = form.cleaned_data['review']
                data.ip = request.META.get('REMOTE_ADDR')
                data.product_id = product_id
                data.user_id = request.user.id
                data.save()
                messages.success(request, 'Thank you, your review Posted!')
                return redirect(url)


def run_python_code(request):
    product_name = request.GET.get('name','')
    shirt_path = os.path.join(settings.BASE_DIR, f'media/photos/products/{product_name}.png')

    cap = cv2.VideoCapture(0)
    detector = PoseDetector(staticMode=False,
                            modelComplexity=1,
                            smoothLandmarks=False,
                            enableSegmentation=False,
                            smoothSegmentation=True,
                            detectionCon=0.5,
                            trackCon=0.5)

    # Path to the shirt image

    imgshirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)

    fixedratio = 262 / 190  # width of shirt / width of points
    shirt_ratio_width_height = 581 / 440


    # Define the face landmark indices to ignore
    face_landmark_indices = list(range(0, 17))

    while True:
        success, img = cap.read()

        # Detect pose
        img = detector.findPose(img, draw=False)

        # Find the landmarks, bounding box, and center of the body in the frame
        # Set draw=True to draw the landmarks and bounding box on the image
        lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

        # Check if any body landmarks are detected
        if lmList:
            lm11 = lmList[11][0:2]
            lm12 = lmList[12][0:2]

            width_of_shirt = int((lm11[0] - lm12[0]) * fixedratio)
            imgshirt_resized = cv2.resize(imgshirt, (width_of_shirt, int(width_of_shirt * shirt_ratio_width_height)))
            currentscale = (lm11[0] - lm12[0]) / 190
            offset = int(44 * currentscale), int(48 * currentscale)

            try:
                img = cvzone.overlayPNG(img, imgshirt_resized, (lm12[0] - offset[0], lm12[1] - offset[1]))
            except:
                pass

        cv2.imshow("Image", img)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break
    return render(request, 'index.html')







