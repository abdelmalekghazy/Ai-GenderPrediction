import cv2 

# Model için ortalama BGR (Mavi, Yeşil, Kırmızı) değerleri
MODEL_MEAN_VALUES = (78.4463377603,
                     87.7689143744,
                     114.895847746)

# Yaş gruplarının listesi
age_list =['(0, 2)','(4, 6)','(8, 12)',
           '(15,20)','(25,32)','(38, 43)','(48, 53)',
           '(60, 100)'
           ]

# Cinsiyet listesi
gender_list =['Male','Female']

# Modelleri yükleyen fonksiyon
def filesGet():
    age_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_age.prototxt',
        'data/age_net.caffemodel'
    )
    gender_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_gender.prototxt',
        'data/gender_net.caffemodel'
    )
    return age_net, gender_net

# Kameradan okuma fonksiyonu
def read_from_camera(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    image = cv2.imread('images/employee.png')  # Görüntüyü okuma
    
    # Yüz algılama için Haar Cascade sınıflandırıcıyı yükleme
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Görüntüyü gri tona çevirme
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)  # Yüzleri algılama
    if len(faces) > 0: 
        print("Found {} Faces".format(str(len(faces))))
    
    for (x, y, w, h) in faces:
        # Algılanan yüzler için dikdörtgen çizme
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
        # Yüz bölgesini kesip blob oluşturma
        face_img = image[y:y + h, h:h + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Cinsiyet tahmini
        gender_net.setInput(blob)
        gender_p = gender_net.forward() 
        gender = gender_list[gender_p[0].argmax()]
        print("Gender : " + gender)
        
        # Yaş tahmini
        age_net.setInput(blob)
        age_p = age_net.forward() 
        age = age_list[age_p[0].argmax()]
        print("Age : " + age)
        
        # Cinsiyet ve yaşı görüntüye yazdırma
        G_A = "%s %s" % (gender, age)
        cv2.putText(image, G_A, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Görüntüyü gösterme
        cv2.imshow('RAKWAN', image)
    
    cv2.waitKey(0)  # Kullanıcıdan herhangi bir tuşa basmasını bekler
    cv2.destroyAllWindows()  # Tüm OpenCV pencerelerini kapatır

# Ana program
if __name__ == "__main__":
    age_net, gender_net = filesGet()  # Modelleri yükleme
    read_from_camera(age_net, gender_net)  # Kameradan okuma ve tahmin yapma
