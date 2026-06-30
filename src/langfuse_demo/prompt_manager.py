from pathlib import Path
from typing import Any

import frontmatter
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError, meta
from utils import get_project_root


class PromptManager:
    def __init__(self, template_dir: str = "prompts/templates"):
        self.template_dir = Path(get_project_root()) / template_dir

        if not self.template_dir.exists():
            raise FileNotFoundError(
                f"Template directory not found: {self.template_dir}"
            )

        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _normalize_template_path(self, template: str) -> str:
        """Normalize template path by ensuring it has the correct extension.

        Args:
            template (str): Template name

        Returns:
            str: Normalized template path
        """
        return f"{template}.j2" if not template.endswith(".j2") else template

    def _load_template_file(self, template_path: str) -> frontmatter.Post:
        """Load and parse template file with frontmatter

        Args:
            template_path (str): Path to template file

        Raises:
            IOError: If template file cannot be loaded

        Returns:
            frontmatter.Post: Parsed template with metadata
        """
        try:
            file_path = self.env.loader.get_source(self.env, template_path)[1]  # type: ignore
            with open(file_path, encoding="utf-8") as f_in:
                return frontmatter.load(f_in)
        except IOError as err:
            raise IOError(f"Failed to load template {template_path}") from err

    def get_prompt(self, template: str, **kwargs: Any) -> str:
        """Render a template with the provided variables

        Args:
            template (str): Template name
            **kwargs: Variables to pass to the template

        Raises:
            ValueError: if template rendering fails

        Returns:
            _type_: _description_
        """
        template_path = self._normalize_template_path(template)
        post = self._load_template_file(template_path)

        try:
            template_obj = self.env.from_string(post.content)
            return template_obj.render(**kwargs)
        except TemplateError as err:
            raise ValueError(f"Error rendering template: {err}") from err

    def get_template_info(self, template: str) -> dict[str, Any]:
        """Get template metadata and variables information

        Args:
            template (str): Template name

        Returns:
            Dict[str, Any]: Dict containing template metadata and placehoders
        """
        template_path = self._normalize_template_path(template)
        post = self._load_template_file(template_path)

        parsed_content = self.env.parse(post.content)
        variables = meta.find_undeclared_variables(parsed_content)
        meta_data = post.metadata

        meta_data["placeholders"] = variables

        return meta_data
