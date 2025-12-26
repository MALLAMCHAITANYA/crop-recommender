from sqlalchemy.orm import Session
from . import models
from .schemas import UserCreate

def create_user(db: Session, user: UserCreate):
    new_user = models.User(
        name=user.name,
        email=user.email,
        password=user.password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()
