import streamlit as st

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="เครื่องคิดเลขจำลอง", page_icon="🧮", layout="wide")

# ส่วนหัวข้อและคำอธิบาย
st.title("🧮 เครื่องคิดเลขจำลอง")
st.caption("ตัวอย่างการใช้ Columns และการคำนวณเบื้องต้น")

# สร้าง 2 คอลัมน์สำหรับกรอกตัวเลข
col1, col2 = st.columns(2)

with col1:
    num1 = st.number_input("กรอกตัวเลขที่ 1:", value=5.00, step=1.00, format="%.2f")

with col2:
    num2 = st.number_input("กรอกตัวเลขที่ 2:", value=5.00, step=1.00, format="%.2f")

# เลือกการดำเนินการ
operation = st.selectbox(
    "เลือกการดำเนินการ:",
    ["บวก (+)", "ลบ (-)", "คูณ (*)", "หาร (/)"]
)

# ปุ่มคำนวณผลลัพธ์
if st.button("คำนวณผลลัพธ์"):
    if operation == "บวก (+)":
        result = num1 + num2
    elif operation == "ลบ (-)":
        result = num1 - num2
    elif operation == "คูณ (*)":
        result = num1 * num2
    elif operation == "หาร (/)":
        if num2 != 0:
            result = num1 / num2
        else:
            result = "ไม่สามารถหารด้วยศูนย์ได้"

    # แสดงผลลัพธ์ในกล่องข้อความสีน้ำเงิน
    if isinstance(result, (int, float)):
        st.info(f"ผลลัพธ์การคำนวณ: {result}")
    else:
        st.error(result)
