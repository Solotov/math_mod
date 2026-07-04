"""
repository.py - Patrón Repository genérico para operaciones CRUD
Abstrae la lógica de BD y permite manipular modelos sin escribir SQL
"""
from typing import TypeVar, Generic, Type, List, Optional, Any
from sqlmodel import Session, select, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect

T = TypeVar("T")  # Tipo genérico para el modelo


class Repository(Generic[T]):
    """
    Repositorio genérico para operaciones CRUD.
    
    Uso:
        repo = Repository[Model](session, Model)
        modelo = repo.get_by_id(1)
        modelos = repo.list_all()
        nuevo = repo.create(ModelCreate(...))
    """
    
    def __init__(self, session: Session, model_class: Type[T]):
        """
        Inicializa el repositorio.
        
        Args:
            session: Sesión de SQLModel
            model_class: Clase del modelo (ej: Model, TrainingSession, etc)
        """
        self.session = session
        self.model_class = model_class
        # Obtener el nombre de la clave primaria mediante inspección SQLAlchemy
        self._pk_name = inspect(self.model_class).primary_key[0].name # type: ignore
    
    @property
    def pk_name(self) -> str:
        """Nombre de la columna de clave primaria."""
        return self._pk_name
    
    # ========================================================================
    # OPERACIONES BÁSICAS CRUD
    # ========================================================================
    
    def create(self, obj: Any) -> T:
        """
        Crea un nuevo registro en la BD.
        
        Args:
            obj: Instancia del modelo o schema de creación
            
        Returns:
            El objeto creado con ID asignado
            
        Raises:
            RuntimeError: Si hay error en la BD
        """
        try:
            # Si ya es una instancia del modelo, la usamos directamente
            if isinstance(obj, self.model_class):
                db_obj = obj
            else:
                # Si tiene método .dict() (Pydantic v1 o schema), lo usamos
                if hasattr(obj, "dict"):
                    db_obj = self.model_class(**obj.dict())
                else:
                    # Asumimos que obj es un dict o compatible
                    db_obj = self.model_class(**obj)
            
            self.session.add(db_obj)
            self.session.commit()
            self.session.refresh(db_obj)
            return db_obj
        except SQLAlchemyError as e:
            self.session.rollback()
            raise RuntimeError(f"Error creando {self.model_class.__name__}: {str(e)}")
    
    def get_by_id(self, id: Any) -> Optional[T]:
        """
        Obtiene un registro por su clave primaria.
        
        Args:
            id: Valor de la clave primaria
            
        Returns:
            El objeto si existe, None si no
        """
        try:
            pk_column = getattr(self.model_class, self.pk_name)
            statement = select(self.model_class).where(pk_column == id)
            return self.session.exec(statement).first()
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error obteniendo {self.model_class.__name__} por {self.pk_name}: {str(e)}")
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """
        Obtiene todos los registros con paginación.
        
        Args:
            skip: Registros a saltar
            limit: Máximo de registros a retornar
            
        Returns:
            Lista de objetos
        """
        try:
            statement = select(self.model_class).offset(skip).limit(limit)
            return list(self.session.exec(statement).all())
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error listando {self.model_class.__name__}: {str(e)}")
    
    def update(self, id: Any, obj_update: Any) -> Optional[T]:
        """
        Actualiza un registro existente.
        
        Args:
            id: Valor de la clave primaria del registro a actualizar
            obj_update: Schema con los campos a actualizar
            
        Returns:
            El objeto actualizado, None si no existe
            
        Raises:
            RuntimeError: Si hay error en la BD
        """
        try:
            db_obj = self.get_by_id(id)
            if not db_obj:
                return None
            
            # Obtener datos de actualización
            if hasattr(obj_update, "dict"):
                update_data = obj_update.dict(exclude_unset=True)
            else:
                # Si es un dict o similar, filtramos None
                update_data = {k: v for k, v in vars(obj_update).items() if v is not None}
            
            for key, value in update_data.items():
                if value is not None:
                    setattr(db_obj, key, value)
            
            self.session.add(db_obj)
            self.session.commit()
            self.session.refresh(db_obj)
            return db_obj
        except SQLAlchemyError as e:
            self.session.rollback()
            raise RuntimeError(f"Error actualizando {self.model_class.__name__}: {str(e)}")
    
    def delete(self, id: Any) -> bool:
        """
        Elimina un registro.
        
        Args:
            id: Valor de la clave primaria del registro a eliminar
            
        Returns:
            True si se eliminó, False si no existe
            
        Raises:
            RuntimeError: Si hay error en la BD
        """
        try:
            db_obj = self.get_by_id(id)
            if not db_obj:
                return False
            
            self.session.delete(db_obj)
            self.session.commit()
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            raise RuntimeError(f"Error eliminando {self.model_class.__name__}: {str(e)}")
    
    # ========================================================================
    # OPERACIONES AVANZADAS
    # ========================================================================
    
    def get_by_field(self, field_name: str, value: Any) -> Optional[T]:
        """
        Obtiene el primer registro que coincida con un campo.
        
        Args:
            field_name: Nombre del campo
            value: Valor a buscar
            
        Returns:
            El objeto si existe, None si no
        """
        try:
            field = getattr(self.model_class, field_name, None)
            if field is None:
                raise ValueError(f"Campo {field_name} no existe en {self.model_class.__name__}")
            
            statement = select(self.model_class).where(field == value)
            return self.session.exec(statement).first()
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error buscando {self.model_class.__name__} por {field_name}: {str(e)}")
    
    def get_all_by_field(self, field_name: str, value: Any) -> List[T]:
        """
        Obtiene todos los registros que coincidan con un campo.
        
        Args:
            field_name: Nombre del campo
            value: Valor a buscar
            
        Returns:
            Lista de objetos
        """
        try:
            field = getattr(self.model_class, field_name, None)
            if field is None:
                raise ValueError(f"Campo {field_name} no existe en {self.model_class.__name__}")
            
            statement = select(self.model_class).where(field == value)
            return list(self.session.exec(statement).all())
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error buscando {self.model_class.__name__} por {field_name}: {str(e)}")
    
    def count(self) -> int:
        """
        Cuenta todos los registros.
        
        Returns:
            Número total de registros
        """
        try:
            pk_column = getattr(self.model_class, self.pk_name)
            statement = select(func.count(pk_column))
            return self.session.exec(statement).one()
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error contando {self.model_class.__name__}: {str(e)}")
    
    def exists(self, id: Any) -> bool:
        """
        Verifica si existe un registro por su clave primaria.
        
        Args:
            id: Valor de la clave primaria
            
        Returns:
            True si existe, False si no
        """
        return self.get_by_id(id) is not None
    
    def delete_all(self) -> int:
        """
        Elimina todos los registros (CUIDADO!).
        
        Returns:
            Número de registros eliminados
        """
        try:
            statement = select(self.model_class)
            objects = self.session.exec(statement).all()
            for obj in objects:
                self.session.delete(obj)
            self.session.commit()
            return len(objects)
        except SQLAlchemyError as e:
            self.session.rollback()
            raise RuntimeError(f"Error eliminando todos en {self.model_class.__name__}: {str(e)}")